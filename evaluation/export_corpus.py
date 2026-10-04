"""Export the synthetic evaluation corpus to real .docx files.

Why this exists
---------------
The corpus lives as Python dicts in evaluation/corpus.py, which is what the
evaluation harness wants: reproducible, diffable, and no binary files in git.
But it means the 32 candidates cannot be searched through the actual service,
because the indexer reads .pdf and .docx files from a folder.

This writes them out so you can drive the real pipeline end to end -- index
them, start the API, and run job descriptions against them by hand.

    python -m evaluation.export_corpus               # writes to ./cvs
    python -m evaluation.export_corpus --out demo    # somewhere else
    python -m evaluation.export_corpus --list        # just print the roster

The files are deliberately named after each candidate, because the indexer
derives a fallback name from the filename. Pass --messy-filenames to export
them with uninformative names instead (cv_final_v2.docx and friends), which
exercises the LLM metadata fallback.
"""

import argparse
from pathlib import Path

from docx import Document

from evaluation.corpus import ALL_QUERIES, CANDIDATES, render_cv


def filename_for(candidate: dict, messy: bool, index: int) -> str:
    if messy:
        # Names the filename cannot help with, so metadata extraction has to
        # read the document body instead.
        patterns = ["cv_final_v2", "resume_copy", "document_updated", "cv", "profile_new"]
        return f"{patterns[index % len(patterns)]}_{candidate['id']}.docx"
    return f"{candidate['name'].replace(' ', '_')}_Resume.docx"


def write_docx(path: Path, text: str) -> None:
    """Write resume text as a .docx, one paragraph per line."""
    doc = Document()
    for line in text.split("\n"):
        doc.add_paragraph(line)
    doc.save(str(path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the synthetic CV corpus as .docx files."
    )
    parser.add_argument("--out", default="cvs", help="output directory (default: ./cvs)")
    parser.add_argument(
        "--messy-filenames", action="store_true",
        help="use uninformative filenames, exercising the LLM metadata fallback",
    )
    parser.add_argument("--list", action="store_true", help="print the roster and exit")
    args = parser.parse_args()

    if args.list:
        print(f"{len(CANDIDATES)} candidates in the corpus:\n")
        for c in CANDIDATES:
            print(f"  {c['id']}  {c['name']:20} {c['title']:28} "
                  f"{c['location']:10} {c['years']:>2}y")
        print(f"\n{len(ALL_QUERIES)} labelled job descriptions:\n")
        for q in ALL_QUERIES:
            strong = [k for k, v in q["relevance"].items() if v == 2]
            partial = [k for k, v in q["relevance"].items() if v == 1]
            print(f"  {q['id']:24} strong={','.join(strong) or '-':22} "
                  f"partial={','.join(partial) or '-'}")
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for i, candidate in enumerate(CANDIDATES):
        path = out / filename_for(candidate, args.messy_filenames, i)
        write_docx(path, render_cv(candidate))

    print(f"Wrote {len(CANDIDATES)} CVs to {out.resolve()}")
    print()
    print("Next:")
    print(f"  python -m indexer.run                    # index them")
    print(f"  python -m uvicorn api.main:app --port 8000")
    print()
    print("Then POST a job description to /api/v1/screen, or open /docs.")
    print("Run with --list to see which candidates the labelled queries expect.")


if __name__ == "__main__":
    main()
