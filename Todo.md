# Technical Discussion Notes

## 1. `requirements.txt` Placement Decision

- Evaluate the optimal location for `requirements.txt` within the project structure to ensure clarity and maintainability.

---

## 2. Integration with SonarQube Using SonarLint

### VS Code Integration

- The **SonarLint** extension already provides inline highlighting, explanations, and one-click “fix suggestion” links (where available).
- The **bootstrap script** can optionally offer to install SonarLint alongside other recommended extensions.
- This approach allows developers to choose whether to accept or ignore suggestions — maintaining a clean, developer-friendly workflow.

---

## 3. Pulling Code Updates Manually

### Why Not Auto-Pull?

Auto-pulling within the formatting script is technically possible (e.g., `git fetch && git pull` before running the tooling), but it introduces several issues:

- **Safety:** Formatting often runs in dirty worktrees; a forced pull could cause merge conflicts or lost work.
- **CI vs Local:** In CI/CD, clean checkouts are guaranteed; locally, developers may intentionally stay pinned to a specific branch or tag.
- **Control:** Combining source updates with formatting couples two independent workflows, reducing developer flexibility.
- **Credentials / Network Issues:** Formatters might run in environments without Git credentials or network access, leading to errors.

---

### Recommended Approach

If pull integration is still desired, a **safer middle ground** is:

1. Add an optional flag `--update-from-origin` to `bootstrap.py` or `format.py` that:

   - Runs `git fetch` / `git pull` when invoked.
   - Displays clear prompts and warnings.
   - Aborts if local modifications are detected.

2. Provide a dedicated script or alias, e.g.:
   - `tools/update_and_format.py`
   - This script would first update the repository (with safety checks) and then call `format.py`.

This keeps the **core formatting automation predictable**, while still offering a convenient one-liner for teams that prefer an “update + format” workflow.
