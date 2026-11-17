# METcalcpy repository conventions

Guidance for AI coding agents working in this repository, and a useful summary
for human contributors.

## Read the Contributor's Guides first

The [METplus Contributor's
Guide](https://metplus.readthedocs.io/en/latest/Contributors_Guide/index.html) covers the process shared by every METplus component.
Its [GitHub Workflow](https://metplus.readthedocs.io/en/latest/Contributors_Guide/github_workflow.html) chapter applies here in full:
how issues, labels, milestones, and project boards are used, how branches are
named, and how pull requests are opened, reviewed, and merged.

The [METcalcpy Contributor's
Guide](https://metplus.readthedocs.io/projects/metcalcpy/en/latest/Contributors_Guide/index.html),
whose source is `docs/Contributors_Guide/index.rst` in this repository, covers
where code belongs, testing, and documentation. METcalcpy follows the [METplus
coding standards](https://metplus.readthedocs.io/en/latest/Contributors_Guide/coding_standards.html).

This file is the short form: the steps that are easy to miss. The guides are
authoritative. Where this file and a guide genuinely conflict, say so in the
pull request rather than silently following one.

## Checks that are easy to miss

- New code needs pytest tests under `test/`, with any sample data under 100 MB
  in `test/data`, and new tests added to `.github/workflows/unit_tests.yaml`.
- Document Python code with docstrings that follow
  [PEP 257](https://peps.python.org/pep-0257/).
- A new Python package goes in `requirements.txt`.

## Pull requests

Fill in `.github/pull_request_template.md`. Change `- [ ]` to `- [x]` on every
item that has been answered or filled in, and leave unanswered items and the
action items under "Pull Request Checklist" unchecked.
