Exo Collection System - Windows x64
===================================

This bundle contains both desktop applications:

1. Run_ExoCollector.cmd
   Opens Exo Collector for device preflight, simulated multimodal collection,
   live preview, synchronization waiting, controlled stop, and Trial finalization.

2. Run_ExoDataStudio.cmd
   Opens Exo Data Studio for finalized-Trial browsing, statistics, offline
   review, recovery tools, external-artifact import, and manual offline upload.

No application command-line parameters are required. On first launch, choose a
data root in the UI. Both applications should use the same data root; the choice
is remembered for the current Windows user.

Real-hardware requirements
--------------------------

The hardware profile supports Raw Ethernet ultrasound through Scapy/Npcap,
three Xsens MTw sensors through the official MT SDK Python binding, and the
documented Teensy serial protocol. Install Npcap with WinPcap API-compatible
Mode before using ultrasound. A frozen build includes Python hardware modules
only when they were installed in the Python environment used to build it.
Physical devices still require a laboratory connection and long-run acceptance
test; a successful software smoke test is not a claim of hardware acceptance.
The synchronization-pulse device in the current hardware profile remains a
bench simulator until the force-plate/motion-capture input hardware is supplied.

Safe operating notes
--------------------

- Default to project T (test) until the complete experiment setup is verified.
- Exo Collector arms first and waits for a qualified synchronization pulse; it
  does not use a fixed interactive acquisition duration.
- Do not copy, replay, checksum, recover, import, or upload large data while a
  collection is active. Data Studio automatically enters lightweight mode when
  it detects the Collector activity lease.
- Only FINALIZED Trial packages may be uploaded. Passwords/private-key
  passphrases are entered in the UI and are not stored.
- Keep BUILD_MANIFEST.json with the bundle when archiving a release. It records
  Git provenance, verification status, exact build versions, and SHA-256 for
  every executable, launcher, and included instruction file. The adjacent
  .zip.sha256 file verifies the complete ZIP before extraction.

For architecture and developer details, see README_PROJECT.md.
