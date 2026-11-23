# Third-party components

| Component | Use | License |
|---|---|---|
| [OpenKeyScan](https://github.com/rekordcloud/openkeyscan-analyzer) (Rekord Cloud) | KeyNet CNN for musical key; CQT front-end ported to JS and Rust | MIT |
| [openkeyscan-reaper](https://github.com/fdrv161/openkeyscan-reaper) (fdrv161) | ONNX conversions of KeyNet and DeepRhythm; reference NumPy implementation of the CQT used to verify the ports | see repository |
| [DeepRhythm](https://github.com/bleugreen/deeprhythm) | Tempo network; HCQM + onset-refinement logic ported to JS (`ui/index.html`) and Rust (`core/src/tempo.rs`) | **AGPL-3.0** |
| [MAEST](https://github.com/palonso/maest) (Alonso-Jiménez et al.; ONNX export `Xenova/discogs-maest-30s-pw-73e-ts`) | Genre/style classification; feature extraction and the 400 Discogs labels ported to Rust (`core/src/maest.rs`, `core/data/`) | code: AGPL-3.0 (see repository); **weights: check the model card — typically CC BY-NC-SA 4.0 (non-commercial)** |
| ONNX Runtime Web, Transformers.js | Inference in the WebView (loaded from jsDelivr) | MIT / Apache-2.0 |
| lofty | Tag reading/writing | MIT / Apache-2.0 |
| tract, symphonia, rustfft, Tauri | Native core (inference, decoding, FFT) and desktop shell | MIT / Apache-2.0 / MPL-2.0 (symphonia) |

Because the UI contains a port of AGPL-3.0 code (DeepRhythm), this project as a whole is distributed under the **AGPL-3.0**
(see `LICENSE`). If you redistribute binaries, you must also offer the corresponding source (a public GitHub repository
satisfies this). The model files themselves keep their original licenses.
