"""Desktop UI for chartgen.

    python -m chartgen.app

Tkinter on purpose: it ships with Python, so the UI adds no dependency and
nothing extra to bundle. Generation runs on a worker thread and reports through a
queue — Tk widgets are only ever touched from the main thread.
"""
import json
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from argparse import Namespace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from . import pipeline

SETTINGS = Path(os.environ.get("APPDATA", Path.home())) / "chartgen" / "settings.json"
MODELS = {
    "Quality (M, ~225M params)": "3podi/charter-v1.0-40-M-best-acc",
    "Fast (S, ~25M params)": "3podi/charter-v1.0-40-S-best-acc",
}
GRIDS = {"16th notes (default)": 4, "8th notes (sparser)": 2, "Triplet 8ths": 3}
# Named from the playtest verdict rather than the implementation: the neural
# model samples, so it "does some things better and some worse — bigger wins
# but bigger losses", while transcription is deterministic and steadier.
ENGINES = {
    "Transcription — steady (default)": "basicpitch",
    "Neural model — bigger highs and lows": "audio2chart",
}
FRET_MODES = {
    "Follow the melody (pitch)": "pitch",
    "Raw model output": "model",
}
# Transcribing vocals invents words over instrumental outros - Whisper falls
# back on the YouTube captions it was trained on, and a real chart ended with
# "Thank you for watching". Looking the song up first avoids the guesswork
# entirely when someone has already synced it.
LYRIC_SOURCES = {
    "Look up online, else listen": "auto",
    "Look up online only": "online",
    "Listen to the vocals": "transcribe",
}


def fmt_elapsed(seconds: float) -> str:
    """42s -> '42s', 754s -> '12m 34s', 5025s -> '1h 23m'."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds // 3600}h {seconds % 3600 // 60}m"


def load_settings() -> dict:
    try:
        return json.loads(SETTINGS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(values: dict) -> None:
    try:
        SETTINGS.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS.write_text(json.dumps(values, indent=2), encoding="utf-8")
    except OSError:
        pass  # settings are a convenience; never fail a run over them


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.events: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.batch_files: list[str] = []
        self.batch_pos: tuple[int, int] | None = None
        self.cancel = threading.Event()
        self.result = None
        self.started = 0.0
        saved = load_settings()

        root.title("chartgen — Clone Hero chart generator")
        root.minsize(720, 560)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(3, weight=1)

        self._build_input(saved)
        self._build_options(saved)
        self._build_actions()
        self._build_output()
        root.after(100, self._drain)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------------------------------------------------------------- layout
    def _build_input(self, saved):
        frame = ttk.LabelFrame(self.root, text="Song", padding=8)
        frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        frame.columnconfigure(1, weight=1)

        self.audio = tk.StringVar()
        ttk.Label(frame, text="Audio / YouTube").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.audio).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(frame, text="Browse…", command=self._pick_audio).grid(row=0, column=2)
        ttk.Label(frame, text="Paste a YouTube link here to download and chart it.",
                  foreground="#666").grid(row=1, column=1, sticky="w", padx=6)

        self.artist = tk.StringVar(value="")
        self.title = tk.StringVar(value="")
        ttk.Label(frame, text="Artist").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(frame, textvariable=self.artist).grid(row=2, column=1, sticky="ew",
                                                       padx=6, pady=(6, 0))
        ttk.Label(frame, text="Title").grid(row=3, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(frame, textvariable=self.title).grid(row=3, column=1, sticky="ew",
                                                      padx=6, pady=(6, 0))

        self.outdir = tk.StringVar(value=saved.get("outdir", str(Path("out").resolve())))
        ttk.Label(frame, text="Save to").grid(row=4, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(frame, textvariable=self.outdir).grid(row=4, column=1, sticky="ew",
                                                       padx=6, pady=(6, 0))
        ttk.Button(frame, text="Browse…", command=self._pick_outdir).grid(
            row=4, column=2, pady=(6, 0))
        ttk.Label(frame, text="Tip: point this at your Clone Hero songs folder.",
                  foreground="#666").grid(row=5, column=1, sticky="w", padx=6)

    def _build_options(self, saved):
        frame = ttk.LabelFrame(self.root, text="Options", padding=8)
        frame.grid(row=1, column=0, sticky="ew", padx=10, pady=4)
        for col in (1, 3):
            frame.columnconfigure(col, weight=1)

        self.engine = tk.StringVar(value=saved.get("engine", list(ENGINES)[0]))
        ttk.Label(frame, text="Note source").grid(row=0, column=0, sticky="w")
        ttk.Combobox(frame, textvariable=self.engine, values=list(ENGINES),
                     state="readonly", width=32).grid(row=0, column=1, sticky="w", padx=6)

        self.grid_choice = tk.StringVar(value=saved.get("grid", list(GRIDS)[0]))
        ttk.Label(frame, text="Note grid").grid(row=0, column=2, sticky="w")
        ttk.Combobox(frame, textvariable=self.grid_choice, values=list(GRIDS),
                     state="readonly", width=20).grid(row=0, column=3, sticky="w", padx=6)

        self.fret_mode = tk.StringVar(value=saved.get("fret_mode", list(FRET_MODES)[0]))
        ttk.Label(frame, text="Fret choice").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Combobox(frame, textvariable=self.fret_mode, values=list(FRET_MODES),
                     state="readonly", width=26).grid(row=1, column=1, sticky="w",
                                                     padx=6, pady=(6, 0))

        self.target_diff = tk.StringVar(value=saved.get("target_diff", "Auto"))
        ttk.Label(frame, text="Max difficulty").grid(row=1, column=2, sticky="w", pady=(6, 0))
        ttk.Combobox(frame, textvariable=self.target_diff,
                     values=["Auto"] + [str(n) for n in range(7)],
                     state="readonly", width=6).grid(row=1, column=3, sticky="w",
                                                    padx=6, pady=(6, 0))

        self.attempts = tk.IntVar(value=saved.get("attempts", 3))
        ttk.Label(frame, text="Retries if poor").grid(row=2, column=2, sticky="w", pady=(6, 0))
        ttk.Spinbox(frame, from_=1, to=8, textvariable=self.attempts, width=5).grid(
            row=2, column=3, sticky="w", padx=6, pady=(6, 0))

        self.sustain_gap = tk.DoubleVar(value=saved.get("sustain_gap", 1.0))
        ttk.Label(frame, text="Sustain gap (beats)").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(frame, from_=0.25, to=4.0, increment=0.25, width=5,
                    textvariable=self.sustain_gap).grid(row=2, column=1, sticky="w",
                                                        padx=6, pady=(6, 0))

        self.model = tk.StringVar(value=saved.get("model", list(MODELS)[0]))
        ttk.Label(frame, text="Neural checkpoint").grid(row=3, column=0, sticky="w",
                                                       pady=(6, 0))
        ttk.Combobox(frame, textvariable=self.model, values=list(MODELS),
                     state="readonly", width=26).grid(row=3, column=1, sticky="w",
                                                      padx=6, pady=(6, 0))
        ttk.Label(frame, text="(only used by the neural source)",
                  foreground="#666").grid(row=3, column=2, columnspan=2, sticky="w")

        self.lyric_source = tk.StringVar(
            value=saved.get("lyric_source", list(LYRIC_SOURCES)[0]))
        ttk.Label(frame, text="Lyrics from").grid(row=4, column=0, sticky="w",
                                                 pady=(6, 0))
        ttk.Combobox(frame, textvariable=self.lyric_source,
                     values=list(LYRIC_SOURCES), state="readonly",
                     width=26).grid(row=4, column=1, sticky="w", padx=6, pady=(6, 0))

        toggles = ttk.Frame(frame)
        toggles.grid(row=5, column=0, columnspan=4, sticky="w", pady=(8, 0))
        self.sustains = tk.BooleanVar(value=saved.get("sustains", True))
        self.star_power = tk.BooleanVar(value=saved.get("star_power", True))
        self.sections = tk.BooleanVar(value=saved.get("sections", True))
        self.hopos = tk.BooleanVar(value=saved.get("hopos", True))
        self.lyrics = tk.BooleanVar(value=saved.get("lyrics", True))
        for text, var in (("Sustains", self.sustains), ("Star power", self.star_power),
                          ("Sections", self.sections), ("HOPOs", self.hopos),
                          ("Lyrics", self.lyrics)):
            ttk.Checkbutton(toggles, text=text, variable=var).pack(side="left", padx=(0, 14))

    def _build_actions(self):
        frame = ttk.Frame(self.root)
        frame.grid(row=2, column=0, sticky="ew", padx=10, pady=4)
        frame.columnconfigure(2, weight=1)
        self.go = ttk.Button(frame, text="Generate chart", command=self._start)
        self.go.grid(row=0, column=0)
        self.stop = ttk.Button(frame, text="Cancel", command=self._request_cancel,
                               state="disabled")
        self.stop.grid(row=0, column=1, padx=6)
        self.bar = ttk.Progressbar(frame, mode="determinate", maximum=6)
        self.bar.grid(row=0, column=2, sticky="ew", padx=6)
        self.status = ttk.Label(frame, text="Ready", width=26, anchor="e")
        self.status.grid(row=0, column=3)

    def _build_output(self):
        frame = ttk.LabelFrame(self.root, text="Progress", padding=8)
        frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=4)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        self.log = tk.Text(frame, height=12, wrap="none", font=("Consolas", 9))
        self.log.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(frame, command=self.log.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scroll.set, state="disabled")

        actions = ttk.Frame(self.root)
        actions.grid(row=4, column=0, sticky="ew", padx=10, pady=(0, 10))
        self.open_button = ttk.Button(actions, text="Open song folder",
                                      command=self._open_folder, state="disabled")
        self.open_button.pack(side="left")

    # ---------------------------------------------------------------- events
    def _pick_audio(self):
        paths = filedialog.askopenfilenames(
            title="Choose songs (at least 30 seconds each; multi-select works)",
            filetypes=[("Audio", "*.mp3 *.flac *.wav *.ogg *.opus *.m4a"), ("All", "*.*")])
        if not paths:
            return
        if len(paths) == 1:
            self.batch_files = []
            self.audio.set(paths[0])
            # Fill artist/title from the file's tags (or its name) so the
            # song shows up properly in CH without any typing. Only fields
            # the user left empty are touched.
            from . import tags

            artist, name = tags.guess_metadata(paths[0])
            if not self.title.get() and name:
                self.title.set(name)
            if not self.artist.get() and artist:
                self.artist.set(artist)
        else:
            # Batch: metadata comes from each file's own tags at run time.
            self.batch_files = list(paths)
            self.audio.set(f"{len(paths)} files selected")
            self.title.set("")
            self.artist.set("")

    def _pick_outdir(self):
        path = filedialog.askdirectory(title="Where to save the song folder")
        if path:
            self.outdir.set(path)

    def _write(self, line):
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _start(self):
        import copy

        from . import youtube

        audio = self.audio.get().strip()
        batch = list(self.batch_files) if audio.endswith("files selected") else []
        # Several links can be pasted at once, space-separated; if every token
        # is a YouTube URL, they queue like a file batch. Playlists expand in
        # the worker (it needs the network).
        tokens = audio.split()
        if not batch and len(tokens) > 1 and all(map(youtube.is_youtube_url, tokens)):
            batch = tokens
        is_url = youtube.is_youtube_url(audio)
        if not batch and (not audio or (not is_url and not Path(audio).is_file())):
            messagebox.showerror(
                "chartgen", "Choose audio file(s) or paste YouTube link(s) first.")
            return
        save_settings(self._settings())
        base = Namespace(
            # URLs must stay strings; Path("https://…") collapses the //.
            audio=audio if is_url else (Path(audio) if not batch else None),
            outdir=Path(self.outdir.get()),
            model=MODELS[self.model.get()], subdiv=GRIDS[self.grid_choice.get()],
            temperature=0.5, top_k=32, attempts=int(self.attempts.get()),
            min_variety=0.80, min_sustain_gap=float(self.sustain_gap.get()),
            hopos=self.hopos.get(), no_star_power=not self.star_power.get(),
            no_sections=not self.sections.get(), no_sustains=not self.sustains.get(),
            engine=ENGINES.get(self.engine.get(), "basicpitch"),
            min_fidelity=0.60,
            fret_mode=FRET_MODES.get(self.fret_mode.get(), "pitch"),
            target_diff=(None if self.target_diff.get() == "Auto"
                         else int(self.target_diff.get())),
            lyrics=self.lyrics.get(),
            lyric_source=LYRIC_SOURCES.get(self.lyric_source.get(), "auto"),
            name=self.title.get().strip() or None,
            artist=self.artist.get().strip() or "Unknown",
            album="", genre="", year="",
        )
        queue_items = batch or [base.audio]

        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.cancel.clear()
        self.result = None
        self.batch_pos = None
        self.started = time.monotonic()
        self.bar["value"] = 0
        self.bar["maximum"] = (7 if is_url else 6) * len(queue_items)
        self.go.configure(state="disabled")
        self.stop.configure(state="normal")
        self.open_button.configure(state="disabled")
        self.status.configure(text="Working…")
        if len(queue_items) > 1:
            self._write(f"Batch of {len(queue_items)} songs — each gets its own "
                        f"folder, named from its tags.\n")
        else:
            self._write("This takes a few minutes. The window stays responsive.\n")

        def work():
            done = failed = 0
            last_result = None
            try:
                try:
                    items = youtube.expand_inputs(
                        queue_items, lambda m: self.events.put(("log", m)))
                except ValueError as error:
                    self.events.put(("error", str(error)))
                    return
                if len(items) != len(queue_items):
                    self.events.put(("max", (7 if is_url else 6) * len(items)))
                def chart_one(item, position, total):
                    opts = copy.copy(base)
                    opts.audio = item
                    if total > 1:
                        self.events.put(("song", (position, total)))
                        self.events.put(("log", f"\n=== [{position}/{total}] "
                                                f"{Path(str(item)).name} ==="))
                        # Per-song metadata from each file's own tags; songs
                        # already charted are skipped, so re-pasting a batch
                        # never redoes finished work.
                        opts.name, opts.artist = None, "Unknown"
                        opts.skip_existing = True
                    return pipeline.run(
                        opts, progress=lambda m: self.events.put(("log", m)),
                        should_cancel=self.cancel.is_set)

                retry = []
                for i, item in enumerate(items, 1):
                    try:
                        result = chart_one(item, i, len(items))
                        if not result.get("skipped"):
                            last_result = result
                            done += 1
                    except (ValueError, OSError) as error:
                        if len(items) == 1:
                            raise
                        self.events.put(("log", f"FAILED: {error}"))
                        retry.append(item)
                # Second chance for songs YouTube 403-blocked mid-batch; the
                # throttle usually lifts while the rest of the queue charts.
                if retry and not self.cancel.is_set():
                    self.events.put(("log", f"\nRetrying {len(retry)} failed…"))
                    for i, item in enumerate(retry, 1):
                        try:
                            result = chart_one(item, i, len(retry))
                            if not result.get("skipped"):
                                last_result = result
                                done += 1
                        except (ValueError, OSError) as error:
                            failed += 1
                            self.events.put(("log", f"FAILED again: {error}"))
                if len(items) > 1:
                    self.events.put(("log", f"\nBatch finished: {done} ok, {failed} failed."))
                self.events.put(("done", last_result))
            except pipeline.Cancelled:
                self.events.put(("cancelled", None))
            except Exception as error:  # surface everything; there is no console
                self.events.put(("error", f"{type(error).__name__}: {error}"))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def _request_cancel(self):
        self.cancel.set()
        self.status.configure(text="Cancelling…")
        self._write("cancelling after the current stage…")

    def _drain(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "max":
                    self.bar["maximum"] = payload
                elif kind == "song":
                    self.batch_pos = payload
                elif kind == "log":
                    self._write(payload)
                    if payload.startswith("["):
                        self.bar["value"] = self.bar["value"] + 1
                elif kind == "done":
                    self._finish(payload)
                elif kind == "cancelled":
                    self._write("\nCancelled.")
                    self._reset("Cancelled")
                elif kind == "error":
                    self._write(f"\nFAILED: {payload}")
                    self._reset("Failed")
                    messagebox.showerror("chartgen", payload)
        except queue.Empty:
            pass
        # The clock lives here, not in the log handler: stages can run for
        # minutes without emitting a line, and the elapsed time should tick
        # every second regardless.
        if self.worker and self.worker.is_alive() and not self.cancel.is_set():
            elapsed = time.monotonic() - self.started
            count = (f"  ·  {self.batch_pos[0]}/{self.batch_pos[1]}"
                     if self.batch_pos else "")
            self.status.configure(text=f"Working… {fmt_elapsed(elapsed)}{count}")
        self.root.after(100, self._drain)

    def _finish(self, result):
        self.result = result
        self.bar["value"] = self.bar["maximum"]
        if result is None or "summary" not in result:
            # Batch where every song failed or was skipped: no table to show.
            self._reset("Done")
            return
        rows = result["summary"]
        self._write("")
        self._write(f"{'tier':<14}{'notes':>7}{'sustains':>10}{'star pow':>10}"
                    f"{'variety':>9}{'walk':>7}")
        for tier, row in rows.items():
            flag = "  <- weak" if min(row["variety"], row["walk"]) < 0.80 else ""
            self._write(f"{tier:<14}{row['notes']:>7}{row['sustains']:>10}"
                        f"{row['star_power']:>10}{row['variety']:>9.2f}"
                        f"{row['walk']:>7.2f}{flag}")
        self._write(f"\nSaved to: {result['song_dir']}")
        self._write("Rescan songs in Clone Hero to see it.")
        self._reset(f"Done in {fmt_elapsed(time.monotonic() - self.started)}")
        self.open_button.configure(state="normal")

    def _reset(self, status):
        self.go.configure(state="normal")
        self.stop.configure(state="disabled")
        self.status.configure(text=status)

    def _open_folder(self):
        if self.result:
            path = str(self.result["song_dir"])
            if sys.platform == "win32":
                os.startfile(path)  # noqa: S606
            else:
                subprocess.Popen(["xdg-open" if sys.platform != "darwin" else "open", path])

    def _settings(self) -> dict:
        return {
            "outdir": self.outdir.get(), "model": self.model.get(),
            "grid": self.grid_choice.get(), "attempts": self.attempts.get(),
            "sustain_gap": self.sustain_gap.get(), "sustains": self.sustains.get(),
            "star_power": self.star_power.get(), "sections": self.sections.get(),
            "hopos": self.hopos.get(), "fret_mode": self.fret_mode.get(),
            "target_diff": self.target_diff.get(), "lyrics": self.lyrics.get(),
            "engine": self.engine.get(),
            "lyric_source": self.lyric_source.get(),
        }

    def _on_close(self):
        if self.worker and self.worker.is_alive():
            if not messagebox.askokcancel("chartgen", "A chart is still generating. Quit?"):
                return
            self.cancel.set()
        # Settings used to persist only after a successful Generate; pointing
        # the output folder somewhere and closing lost it. Closing saves too.
        save_settings(self._settings())
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
