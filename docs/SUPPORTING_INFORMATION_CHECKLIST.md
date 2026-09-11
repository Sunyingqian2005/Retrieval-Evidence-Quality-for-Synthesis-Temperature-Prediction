# Supporting Information and repository checklist

Before submission, verify the following items.

- The manuscript Data and Software Availability statement contains the final
  GitHub or archival DOI URL, not a placeholder.
- The draft text in `MANUSCRIPT_AVAILABILITY_TEXT.md` has been updated and
  copied into the manuscript where appropriate.
- The Associated Content paragraph names the separate Supporting Information
  PDF and briefly describes its contents.
- The Supporting Information contains expanded methods, complete prompts,
  supplementary tables/figures, and enough provenance to connect them to the
  scripts in this repository.
- All prompt templates match the submitted manuscript wording.
- The repository release is archived with Zenodo or another DOI-issuing service
  if a permanent software citation is desired.
- API keys, `.env`, full copyrighted datasets, prompts containing unpublished
  data, and large transient checkpoints are not tracked by Git.
- `python scripts/check_repository.py` passes.
- `python -m unittest discover -s tests -v` passes in the clean environment.
- The author list, license holder, repository URL, version, and eventual article
  DOI in `CITATION.cff` have been checked by all authors.
