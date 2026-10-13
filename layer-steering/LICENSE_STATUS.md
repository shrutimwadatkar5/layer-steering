# Licensing status — READ BEFORE REDISTRIBUTING

This directory is a research project that **depends on** upstream
[goodfire-ai/causalab](https://github.com/goodfire-ai/causalab). Its licensing
status was checked on 2026-10-12 and is reported here rather than resolved.

## What upstream actually says

Checked at commit `8e8d5d1f8f9ca8f42bfce7c6b5eef194f012660e` (2026-09-30):

| Check | Result |
|---|---|
| `LICENSE`, `COPYING`, `NOTICE` in the repository root | **none present** |
| `license` field in `pyproject.toml` | **absent** |
| Any SPDX identifier anywhere in the tree | **none found** |
| `license` object on the GitHub repository API | **absent** |

Under the Berne Convention, an unlicensed work is **all rights reserved by
default**. Absent an explicit grant, the upstream authors retain all rights and
no permission to copy, modify, or redistribute is conferred.

## Consequence for this project

**No license file has been added here, deliberately.**

Adding MIT or Apache-2.0 to this directory would be actively wrong on two
counts:

1. It cannot grant rights this project does not hold. The permission to
   redistribute would flow from the upstream license, not from a choice made
   here.
2. It would misrepresent the terms under which anyone may use the combined work.

Copying an upstream license file would be equally wrong — there is none to copy,
and inventing one would fabricate a grant that was never made.

## What this means in practice

* **Internal research and review use:** fine. Cloning upstream and running this
  directory modifies nothing upstream and is the documented workflow.
* **Public redistribution of this directory as a standalone repository:**
  **do not do it without upstream clarification.** That would require
  permission covering the derivative work, which upstream has not granted.
* **Contributing upstream:** the intended path. The reusable pieces are already
  isolated — see "Upstream contribution candidates" in the README. A pull
  request to `goodfire-ai/causalab` is governed by that project's terms, and
  contributions there are the correct way to share this work broadly.

## Recommended next step

Contact the upstream maintainers (`goodfire-ai`) and ask which license they
intend for `causalab`. Record the answer here, add the matching `LICENSE` file
**and** the `license` field in any packaging metadata only after that
clarification, and keep the upstream commit pin.

## Attribution

This project reuses and builds on:

* **Causalab** — Atticus Geiger, Max Loeffler, and contributors.
  <https://github.com/goodfire-ai/causalab>. License: **none stated**
  (see above). Pinned at `8e8d5d1f8f9ca8f42bfce7c6b5eef194f012660e`.
* **Qwen2.5-1.5B-Instruct** — Alibaba Cloud. Check the model card for its own
  license before redistribution; weights are **not** included in this repository.
* The `MCQA` task, the demo protocol documents, and the on-device execution
  engine are all upstream Causalab code, used unmodified.