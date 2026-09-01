# IEEE Access release checklist

This checklist is for the repository and its computational supplement. It is
not a statement that the manuscript has been accepted.

## Before the first commit

- [ ] Confirm the author name, affiliation and preferred contact address for
      Git metadata and `CITATION.cff`.
- [ ] Choose a code license. Third-party data retain their own source terms.
- [ ] Confirm `git status` contains no virtual environment, token, password,
      local outline, or unreviewed raw data file.
- [ ] Run the static release audit and the existing independent checks.
- [ ] Record the exact commit hash and create the immutable tag `v1.0.0`.

## GitHub Release

- [ ] Create or select one public repository containing this unified project.
- [ ] Push the commit and tag; do not upload access tokens in files or URLs.
- [ ] Create a Release from tag `v1.0.0`.
- [ ] Paste the contents of `RELEASE_NOTES_v1.0.0.md` into the release notes.

## Zenodo

- [ ] Connect the GitHub repository to Zenodo using the account owner’s
      authenticated session.
- [ ] Select the exact GitHub Release, not a moving branch.
- [ ] Check title, creators, license, data-attribution text and version.
- [ ] Publish the archive and record both the version DOI and the concept DOI.
- [ ] Add the DOI to the repository citation metadata in a follow-up commit;
      do not rewrite the already archived tag.

## Manuscript disclosure

- [ ] Cite the GitHub repository and Zenodo DOI in the data/code availability
      statement.
- [ ] State that IEEE Access is fully open access and confirm the current APC
      and author instructions before submission.
- [ ] Report D16 numerical deviations and the CAS directional non-replication
      honestly; do not describe the workflow as universally validated.
