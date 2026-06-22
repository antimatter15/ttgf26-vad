![](../../workflows/gds/badge.svg) ![](../../workflows/docs/badge.svg) ![](../../workflows/test/badge.svg) ![](../../workflows/fpga/badge.svg)

# PDM Voice Activity Detector

This is a starter Tiny Tapeout GF180 Verilog project created from the official `TinyTapeout/ttgf-verilog-template` for the current GF26b open shuttle.

- [Read the project documentation](docs/info.md)

## What is Tiny Tapeout?

Tiny Tapeout is an educational project that aims to make it easier and cheaper than ever to get your digital and analog designs manufactured on a real chip.

To learn more and get started, visit https://tinytapeout.com.

## Demo Design

The demo is a one-tile PDM microphone activity-score estimator with a
hardware-friendly spectral scanner:

- `ui[0]`: PDM data bit
- `ui[1]`: sample enable
- `uo[7:0]`: smoothed activity score

The top module is `tt_um_antimatter15_pdm_vad` in [src/project.v](src/project.v). The circuit counts 64-bit PDM density windows into roughly 10 ms frames, mixes those density samples against three square-wave spectral probes near 500 Hz, 562.5 Hz, and 1.5 kHz, combines the band response with frame energy and frame-to-frame energy change, and outputs an 8-bit activity score. The cocotb test in [test/test.py](test/test.py) covers silence, high-energy biased frames, sample-enable hold behavior, mixer response, and score decay after quiet frames.

The GitHub action will automatically build the ASIC files using [LibreLane](https://www.zerotoasiccourse.com/terminology/librelane/).

## Enable GitHub actions to build the results page

- [Enabling GitHub Pages](https://tinytapeout.com/faq/#my-github-action-is-failing-on-the-pages-part)

## Resources

- [FAQ](https://tinytapeout.com/faq/)
- [Digital design lessons](https://tinytapeout.com/digital_design/)
- [Learn how semiconductors work](https://tinytapeout.com/siliwiz/)
- [Join the community](https://tinytapeout.com/discord)
- [Build your design locally](https://www.tinytapeout.com/guides/local-hardening/)

## What next?

- [Submit your design to the next shuttle](https://app.tinytapeout.com/).
- Edit [this README](README.md) and explain your design, how it works, and how to test it.
- Share your project on your social network of choice:
  - LinkedIn [#tinytapeout](https://www.linkedin.com/search/results/content/?keywords=%23tinytapeout) [@TinyTapeout](https://www.linkedin.com/company/100708654/)
  - Mastodon [#tinytapeout](https://chaos.social/tags/tinytapeout) [@matthewvenn](https://chaos.social/@matthewvenn)
  - X (formerly Twitter) [#tinytapeout](https://twitter.com/hashtag/tinytapeout) [@tinytapeout](https://twitter.com/tinytapeout)
  - Bluesky [@tinytapeout.com](https://bsky.app/profile/tinytapeout.com)
