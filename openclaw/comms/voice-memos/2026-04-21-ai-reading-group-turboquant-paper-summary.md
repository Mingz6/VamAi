# AI Reading Group - TurboQuant Paper - 2026-04-21

**Duration:** 1 hour 47 minutes
**Recorded:** 2026-04-21 18:18 local time
**Topic:** Group walkthrough of the **TurboQuant** paper - KV cache quantization for LLMs
**Host:** AI engineer/founder, also UBC adjunct prof for first-year engineering, also teaches salsa at Baza (8:30pm)
**Recorded by:** Ming ("Min") - disclosed his own MeetMate prototype at the end
**Transcription:** whisper.cpp `large-v3-turbo` with **Silero VAD enabled** (primary). VAD pre-screens silence so turbo doesn't hallucinate loops. Cross-checked against `medium.en`. Original turbo run (no VAD) had a 460-line silence-loop and was discarded.

---

## Format

Round-the-table AI reading group (~10-14 people). Quick intros with a 1-10 self-rated AI readiness score, then a paper deep-dive driven by the host with frequent group questions. Conversational, lots of overlapping voices and laughter.

## Attendees (self-introduced, with self-rated AI knowledge)

| Name | Self-rating | Note |
|---|---|---|
| Farah Hussain | 3 | - |
| Mullen | 5 | - |
| Min (Ming) | 8 | Developer; FOMO-driven; recorder of this audio |
| Thomas | 4-6 | Coming back to CS |
| (unnamed, "free") | 3 | Just learning AI |
| Cyrus | 4 | Catching up |
| Rachel | 2 | "Same reason" - learning |
| Noah | 7 | Worked with Pranav last year on transformer architecture |
| Zen | 6 | Also worked with Pranav; engineering, less AI-focused |
| Dom | ~3 | Mentioned mental-health classification interest |
| Sahil | 6 | Heavy AI user at work |
| Anthony | (retired) | Studied CS at UBC in the '80s; bank systems engineer; brought up Shannon's information theory as foundational |
| Murillo | 9 | "Every day I'm in some AI stuff" - passionate self-learner |
| Juan | 7 | Applies AI day-to-day at work |
| Geva | < 0 | Self-deprecating; "definitely far into negatives" |
| Melvi | 6-7 | Worked in AI before, current role isn't |

## Topics Covered (in order)

1. **Intros + AI readiness scoring** (~10 min)
2. **Fundamentals crash course** (~15 min)
	- Vectors (3D analogy, scaling to N-dim in GPT)
	- Dot product / cosine similarity = "how similar are two concepts"
	- Vector embeddings - text -> tokens -> vectors via embedding model
	- Showed a 3Blue1Brown clip on embeddings encoding semantic meaning (king - man + woman ~= queen)
	- Attention - "taking info from past tokens to inform the current one"
	- KV cache - caches past attention computations; grows quadratically
3. **Side digression: GPU/RAM market** (~5 min)
	- DRAM/GDDR prices have surged 3-4x since early 2025 because Samsung, Micron etc. switched production toward HBM (high-bandwidth memory) for AI data centers
	- Anthony framed why this matters for KV cache: it lives in GPU memory, hardware-bound
4. **Shannon information theory** (~5 min)
	- Anthony walked through it (1920s, Morse code analogy, encoding info efficiently)
	- Host's compression analogy: a file of all "A"s compresses massively (predictable); random data doesn't
5. **TurboQuant walkthrough - the core of the meeting** (~50 min)
	- Two-stage: (1) MSE-optimal vector quantizer with random rotation + Lloyd-Max, (2) one-bit residual quantizer for inner-product preservation
	- Step-by-step with a 3D toy vector `(1, 0, 0)`:
	  - Random rotation - why? to flatten "spiky" vectors so quantization loses less
	  - 1-bit quantization to +-0.7 - snap-to-grid analogy (image with 2 vs 64 colors)
	  - Compute residual `R = y - y_hat`
	  - Tiny sketch of residual via 1-bit projections (positive = aligned with random view, negative = away)
	  - Reconstruct similarity score: `y.q ~= (R + y_hat).q` - open the algebra to recover ~similarity
	- "Blurry face + footnotes" intuition: quantized vector = blurry image, residual bits = a few descriptive lines
	- Online vs offline quantization - TurboQuant is online
6. **Experiments section** (~10 min)
	- "Needle in haystack" test on long context - TurboQuant nearly matched full-precision baseline
	- Caveat raised by the group: paper used different/better TPUs+hardware for TurboQuant runs vs baselines - possible methodological asterisk
	- Ran on Nvidia A100
7. **"What would it take to ship this?"** (~5 min)
	- Group consensus: hardware-level. GPUs are physically architected around specific bit widths, so adopting 1-bit residual quantization at scale needs hardware redesign, not just a kernel rewrite
8. **Closing + next session** (~5 min)
	- Host pitched a future session on **AI for biology / mRNA vaccine design** ("OpenVax", the Twitter dog-vaccine story, Brian Johnson immortality joke)
	- Recommended a math-for-AI book (title not captured - author "Anil"?) starting from Perceptron (1960s) up through modern architectures
9. **Post-meeting (last ~2 min) - Ming reveals MeetMate**
	- Someone asks "are you working this whole time?" - Ming: "No, I just locked it the whole meeting and using the LLM to translate into the meeting row"
	- "I built my own [transcriber]" - group reaction: "Dude, it's fucking insane"

## Key Concepts Worth Remembering

- **KV cache** = key-value cache of attention computations from previous tokens. Big bottleneck for LLM inference; grows quadratically; lives in GPU memory.
- **Quantization** = snapping continuous values to a fixed grid (e.g., 1-bit = 2 values). Trade memory for some precision loss.
- **TurboQuant's contribution** = randomly rotate first (so distribution is well-behaved -> beta-distributed coordinates), quantize MSE-optimally, then add a 1-bit residual sketch that preserves *inner product* similarity rather than the full vector.
- **Why preserve inner product, not the vector itself?** Because the downstream task is similarity search, not reconstruction. You don't need a clear face - just enough features to find the right person in the room.

## Action Items / Follow-ups

| # | Owner | Action |
|---|---|---|
| 1 | Host | Schedule the bio + AI / mRNA vaccine reading group session ("two to four weeks from now") |
| 2 | Group | Volunteer to host a future paper if interested - explicit open invite |
| 3 | Ming | Keep building MeetMate prototype (group expressed strong interest) |
| 4 | Anyone curious | Watch the 3Blue1Brown attention/transformer chapter the host referenced |
| 5 | Anyone curious | Read "Attention Is All You Need" (mentioned as foundational) |

## Quotes Worth Keeping

> "The only difference between me and you is I just spend more time. That's basically it."
> - Host, opening framing

> "AI is my escapism."
> - Murillo, on rating himself ~10/10 in AI engagement

> "Right after this I teach a dance class at Baza, 8:30. So if you ever do salsa or bachata, come."
> - Host, end of intro

> "Dude, it's fucking insane."
> - Group reaction to learning Ming built his own meeting transcriber

## Red Flags / Things Worth a Closer Look

- **Methodological asterisk in TurboQuant experiments**: group noticed the TurboQuant runs apparently used better TPUs/hardware than baselines. Worth verifying against the actual paper before citing the perfect-needle-in-haystack result.
- **Whisper silence-hallucination** (resolved): original `large-v3-turbo` run produced a long loop around a quiet stretch. Re-ran with `--vad --vad-model ggml-silero-v5.1.2.bin` and got a clean transcript that also captured 3 names medium.en missed (Juan, Geva, Melvi). VAD is now the default for any long Voice Memo.
- **Residual mini-loops**: one small "It's a good skill" loop (~15 lines) survived VAD around the intro round. Minor.
- **Cross-talk gaps**: the deeper math discussion (TurboQuant steps 4-7) has overlapping voices. Summary above is best-effort reconstruction.

## Files

- `2026-04-21-ai-reading-group-turboquant-paper.qta` - original Voice Memo copy
- `2026-04-21-ai-reading-group-turboquant-paper.m4a` - extracted playable audio
- `2026-04-21-ai-reading-group-turboquant-paper-whisper.txt` / `.srt` - **PRIMARY**: large-v3-turbo + Silero VAD
- `2026-04-21-ai-reading-group-turboquant-paper-crosscheck.txt` / `.srt` - medium.en cross-check
- `2026-04-21-ai-reading-group-turboquant-paper-turbo-no-vad.txt` / `.srt` - first attempt without VAD, kept as a failure reference
