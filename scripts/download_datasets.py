#!/usr/bin/env python3
"""
Download datasets for SymAgent.

Downloads and organizes the following benchmarks from official sources:
  - WebQSP: https://github.com/ksenon/GRAN or https://github.com/tautree/TreeQA
  - CWQ: https://github.com/YoungNLP/CWQ
  - GrailQA: https://github.com/grailqa/grailqa
  - Freebase KG subset (bundled with WebQSP/CWQ)

Usage:
  python scripts/download_datasets.py --datasets webqsp cwq grailqa --data_dir data
  python scripts/download_datasets.py --datasets all --data_dir data
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Official source repositories as specified
DATASET_SOURCES = {
    "webqsp": {
        "repo": "https://github.com/ksenon/GRAN.git",
        "alt_repo": "https://github.com/tautree/TreeQA.git",
        "description": "WebQuestionsSP dataset with Freebase subgraph",
        "expected_files": [
            "train.json", "dev.json", "test.json",
            "entity2id.txt", "relation2id.txt", "freebase_triples.txt",
        ],
    },
    "cwq": {
        "repo": "https://github.com/YoungNLP/CWQ.git",
        "description": "Complex WebQuestions dataset with multi-hop questions",
        "expected_files": [
            "ComplexWebQuestions_train.json",
            "ComplexWebQuestions_dev.json",
            "ComplexWebQuestions_test.json",
            "entity2id.txt", "relation2id.txt", "freebase_triples.txt",
        ],
    },
    "grailqa": {
        "repo": "https://github.com/grailqa/grailqa.git",
        "description": "GrailQA benchmark for KBQA",
        "expected_files": [
            "grailqa_v1.0_train.jsonl",
            "grailqa_v1.0_dev.jsonl",
            "grailqa_v1.0_test.jsonl",
        ],
    },
    "metaqa": {
        "repo": "https://github.com/yuyinz/MetaQA.git",
        "description": "MetaQA multi-hop QA benchmark",
        "expected_files": [
            "kb.txt",
            "qa_train_1hop.txt", "qa_train_2hop.txt", "qa_train_3hop.txt",
            "qa_test_1hop.txt", "qa_test_2hop.txt", "qa_test_3hop.txt",
            "qa_dev_1hop.txt", "qa_dev_2hop.txt", "qa_dev_3hop.txt",
        ],
    },
}


def run_cmd(cmd: list[str], cwd: str | None = None) -> int:
    """Run a shell command and return exit code."""
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(
            "Command failed (exit %d):\n  stdout: %s\n  stderr: %s",
            result.returncode,
            result.stdout[:500],
            result.stderr[:500],
        )
    return result.returncode


def clone_repo(repo_url: str, target_dir: str) -> bool:
    """Clone a git repository with --depth 1 for efficiency.

    Returns:
        True if clone succeeded or directory already exists.
    """
    if os.path.exists(target_dir):
        logger.info("Repository already exists: %s", target_dir)
        return True

    os.makedirs(os.path.dirname(target_dir), exist_ok=True)
    rc = run_cmd(["git", "clone", "--depth", "1", repo_url, target_dir])
    return rc == 0


def _copy_data_files(
    src_dir: str,
    target_dir: str,
    search_subdirs: list[str] | None = None,
    extensions: tuple[str, ...] = (".json", ".txt", ".tsv", ".csv", ".jsonl"),
) -> list[str]:
    """Copy data files from cloned repo to target directory.

    Args:
        src_dir: Root of cloned repository.
        target_dir: Destination directory.
        search_subdirs: Subdirectories to search. If None, search all.
        extensions: File extensions to include.

    Returns:
        List of copied file names.
    """
    os.makedirs(target_dir, exist_ok=True)
    copied = []

    # Determine directories to search
    if search_subdirs:
        search_dirs = [os.path.join(src_dir, d) for d in search_subdirs]
        search_dirs = [d for d in search_dirs if os.path.exists(d)]
    else:
        search_dirs = [src_dir]

    if not search_dirs:
        # Fallback: walk the whole repo but skip .git
        search_dirs = [src_dir]

    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
        for root, dirs, files in os.walk(search_dir):
            # Skip .git directory
            dirs[:] = [d for d in dirs if d != ".git"]
            for fname in files:
                if fname.endswith(extensions):
                    src_file = os.path.join(root, fname)
                    dst_file = os.path.join(target_dir, fname)
                    if not os.path.exists(dst_file):
                        shutil.copy2(src_file, dst_file)
                        copied.append(fname)
                        logger.info("  Copied: %s", fname)

    return copied


def download_webqsp(data_dir: str) -> None:
    """Download WebQSP dataset from GRAN repository.

    The GRAN repo (https://github.com/ksenon/GRAN) contains WebQSP data
    along with a Freebase subgraph. Falls back to TreeQA if GRAN is
    unavailable.

    Expected files in data/webqsp/:
      train.json, dev.json, test.json,
      entity2id.txt, relation2id.txt, freebase_triples.txt
    """
    target = os.path.join(data_dir, "webqsp")
    raw_dir = os.path.join(data_dir, "_raw", "GRAN")

    logger.info("=== Downloading WebQSP dataset ===")

    # Try GRAN first
    success = clone_repo(DATASET_SOURCES["webqsp"]["repo"], raw_dir)
    if not success:
        logger.warning("GRAN clone failed, trying TreeQA alternative...")
        raw_dir = os.path.join(data_dir, "_raw", "TreeQA")
        success = clone_repo(DATASET_SOURCES["webqsp"]["alt_repo"], raw_dir)

    if success:
        # Search common subdirectory names in GRAN/TreeQA
        copied = _copy_data_files(
            raw_dir, target,
            search_subdirs=["data", "data/WebQSP", "WebQSP", "dataset"],
        )
        if copied:
            logger.info("WebQSP: copied %d files", len(copied))
        else:
            # Try copying everything at top level
            _copy_data_files(raw_dir, target)
    else:
        logger.error(
            "Failed to clone WebQSP source. Please download manually from:\n"
            "  - https://github.com/ksenon/GRAN\n"
            "  - https://github.com/tautree/TreeQA\n"
        )

    _create_placeholders(target, ["train.json", "dev.json", "test.json"])


def download_cwq(data_dir: str) -> None:
    """Download CWQ dataset from YoungNLP/CWQ repository.

    Expected files in data/cwq/:
      ComplexWebQuestions_train.json, ComplexWebQuestions_dev.json,
      ComplexWebQuestions_test.json, entity2id.txt, relation2id.txt,
      freebase_triples.txt
    """
    target = os.path.join(data_dir, "cwq")
    raw_dir = os.path.join(data_dir, "_raw", "CWQ")

    logger.info("=== Downloading CWQ dataset ===")

    success = clone_repo(DATASET_SOURCES["cwq"]["repo"], raw_dir)
    if success:
        copied = _copy_data_files(
            raw_dir, target,
            search_subdirs=["dataset", "data", "ComplexWebQuestions"],
        )
        if copied:
            logger.info("CWQ: copied %d files", len(copied))
        else:
            _copy_data_files(raw_dir, target)
    else:
        logger.error(
            "Failed to clone CWQ. Please download manually from:\n"
            "  - https://github.com/YoungNLP/CWQ\n"
        )

    _create_placeholders(target, ["train.json", "dev.json", "test.json"])


def download_grailqa(data_dir: str) -> None:
    """Download GrailQA dataset from grailqa/grailqa repository.

    Expected files in data/grailqa/:
      grailqa_v1.0_train.jsonl, grailqa_v1.0_dev.jsonl,
      grailqa_v1.0_test.jsonl
    """
    target = os.path.join(data_dir, "grailqa")
    raw_dir = os.path.join(data_dir, "_raw", "grailqa")

    logger.info("=== Downloading GrailQA dataset ===")

    success = clone_repo(DATASET_SOURCES["grailqa"]["repo"], raw_dir)
    if success:
        copied = _copy_data_files(
            raw_dir, target,
            search_subdirs=["data", "release", "grailqa_v1.0"],
        )
        if copied:
            logger.info("GrailQA: copied %d files", len(copied))
        else:
            _copy_data_files(raw_dir, target)
    else:
        logger.error(
            "Failed to clone GrailQA. Please download manually from:\n"
            "  - https://github.com/grailqa/grailqa\n"
        )


def download_metaqa(data_dir: str) -> None:
    """Download MetaQA dataset from yuyinz/MetaQA repository.

    Expected files in data/metaqa/:
      kb.txt, qa_train_1hop.txt, qa_test_3hop.txt, etc.
    """
    target = os.path.join(data_dir, "metaqa")
    raw_dir = os.path.join(data_dir, "_raw", "MetaQA")

    logger.info("=== Downloading MetaQA dataset ===")

    success = clone_repo(DATASET_SOURCES["metaqa"]["repo"], raw_dir)
    if success:
        copied = _copy_data_files(
            raw_dir, target,
            search_subdirs=["data", "MetaQA", "dataset"],
        )
        if copied:
            logger.info("MetaQA: copied %d files", len(copied))
        else:
            _copy_data_files(raw_dir, target)
    else:
        logger.error(
            "Failed to clone MetaQA. Please download manually from:\n"
            "  - https://github.com/yuyinz/MetaQA\n"
        )

    _create_placeholders(target, ["kb.txt", "qa_test_3hop.txt"])


def download_freebase(data_dir: str) -> None:
    """Download/organize Freebase KG subset.

    Strategy:
    1. Check if Freebase data already exists from WebQSP/CWQ downloads.
    2. Try cloning IDEA-FinAI/ToG repo which includes Freebase subgraph files.
    3. Try cloning RManLuo/reasoning-on-graphs for additional sources.
    4. If no direct download, offer to extract subgraphs from WebQSP/CWQ QA files.
    """
    fb_dir = os.path.join(data_dir, "freebase")
    os.makedirs(fb_dir, exist_ok=True)

    logger.info("=== Downloading Freebase KG subgraph ===")

    # Check if all required files already exist
    required_files = ["entity2id.txt", "relation2id.txt", "freebase_triples.txt"]
    if all(os.path.exists(os.path.join(fb_dir, f)) for f in required_files):
        logger.info("Freebase data already exists in %s", fb_dir)
        return

    # Step 1: Check if Freebase data exists from WebQSP/CWQ downloads
    found_files = _collect_freebase_from_datasets(data_dir, fb_dir)

    # Step 2: If still missing files, try IDEA-FinAI/ToG repo
    missing = [f for f in required_files if not os.path.exists(os.path.join(fb_dir, f))]
    if missing:
        _try_download_from_tog(data_dir, fb_dir, missing)

    # Step 3: If still missing, try reasoning-on-graphs
    missing = [f for f in required_files if not os.path.exists(os.path.join(fb_dir, f))]
    if missing:
        _try_download_from_rog(data_dir, fb_dir, missing)

    # Step 4: If still missing, offer subgraph extraction
    missing = [f for f in required_files if not os.path.exists(os.path.join(fb_dir, f))]
    if missing:
        logger.info(
            "Could not find all Freebase files. Missing: %s\n"
            "You can extract a Freebase subgraph from WebQSP/CWQ QA files:\n"
            "  python scripts/prepare_kg.py --dataset webqsp "
            "--input_dir data/webqsp --output_dir data/freebase\n"
            "  python scripts/prepare_kg.py --dataset cwq "
            "--input_dir data/cwq --output_dir data/freebase",
            missing,
        )

    # Report final status
    for f in required_files:
        path = os.path.join(fb_dir, f)
        if os.path.exists(path):
            logger.info("  Found: %s", path)
        else:
            logger.warning("  Missing: %s", path)


def _collect_freebase_from_datasets(data_dir: str, fb_dir: str) -> list[str]:
    """Collect Freebase files from WebQSP/CWQ dataset directories."""
    found_files = []
    for ds in ["webqsp", "cwq"]:
        ds_dir = os.path.join(data_dir, ds)
        if not os.path.exists(ds_dir):
            continue
        for fname in os.listdir(ds_dir):
            lower = fname.lower()
            if "freebase" in lower or "entity2id" in lower or "relation2id" in lower:
                src = os.path.abspath(os.path.join(ds_dir, fname))
                dst = os.path.join(fb_dir, fname)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    logger.info("  Copied: %s -> %s", src, dst)
                found_files.append(fname)
    return found_files


def _try_download_from_tog(data_dir: str, fb_dir: str, missing: list[str]) -> None:
    """Try downloading Freebase files from IDEA-FinAI/ToG repository."""
    logger.info("Trying IDEA-FinAI/ToG repository for Freebase data...")
    raw_dir = os.path.join(data_dir, "_raw", "ToG")

    success = clone_repo("https://github.com/IDEA-FinAI/ToG.git", raw_dir)
    if success:
        # ToG stores Freebase data under data/freebase/ or similar
        search_dirs = [
            os.path.join(raw_dir, "data"),
            os.path.join(raw_dir, "freebase"),
            os.path.join(raw_dir, "dataset"),
            raw_dir,
        ]
        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue
            for root, dirs, files in os.walk(search_dir):
                dirs[:] = [d for d in dirs if d != ".git"]
                for fname in files:
                    if fname in missing and fname.endswith(".txt"):
                        src = os.path.join(root, fname)
                        dst = os.path.join(fb_dir, fname)
                        if not os.path.exists(dst):
                            shutil.copy2(src, dst)
                            logger.info("  Copied from ToG: %s", fname)
                            missing.remove(fname)


def _try_download_from_rog(data_dir: str, fb_dir: str, missing: list[str]) -> None:
    """Try downloading Freebase files from RManLuo/reasoning-on-graphs repository."""
    logger.info("Trying reasoning-on-graphs repository for Freebase data...")
    raw_dir = os.path.join(data_dir, "_raw", "reasoning-on-graphs")

    success = clone_repo(
        "https://github.com/RManLuo/reasoning-on-graphs.git", raw_dir
    )
    if success:
        search_dirs = [
            os.path.join(raw_dir, "data"),
            os.path.join(raw_dir, "freebase"),
            raw_dir,
        ]
        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue
            for root, dirs, files in os.walk(search_dir):
                dirs[:] = [d for d in dirs if d != ".git"]
                for fname in files:
                    if fname in missing and fname.endswith(".txt"):
                        src = os.path.join(root, fname)
                        dst = os.path.join(fb_dir, fname)
                        if not os.path.exists(dst):
                            shutil.copy2(src, dst)
                            logger.info("  Copied from ROG: %s", fname)
                            missing.remove(fname)


def _create_placeholders(directory: str, filenames: list[str]) -> None:
    """Create empty placeholder files if they don't exist."""
    os.makedirs(directory, exist_ok=True)
    for fname in filenames:
        fpath = os.path.join(directory, fname)
        if not os.path.exists(fpath):
            if fname.endswith(".json"):
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write("[]")
            else:
                with open(fpath, "w", encoding="utf-8") as f:
                    pass
            logger.info("  Created placeholder: %s", fname)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download datasets for SymAgent from official sources",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["webqsp", "cwq", "grailqa", "metaqa", "freebase", "all"],
        default=["all"],
        help="Datasets to download (default: all)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Root data directory (default: data)",
    )

    args = parser.parse_args()

    datasets = args.datasets
    if "all" in datasets:
        datasets = ["webqsp", "cwq", "grailqa", "metaqa", "freebase"]

    dispatch = {
        "webqsp": download_webqsp,
        "cwq": download_cwq,
        "grailqa": download_grailqa,
        "metaqa": download_metaqa,
        "freebase": download_freebase,
    }

    for ds in datasets:
        logger.info("=" * 50)
        dispatch[ds](args.data_dir)

    logger.info("\nDownload complete. Data directory: %s", os.path.abspath(args.data_dir))
    logger.info(
        "\nNext steps:\n"
        "  1. Verify data files are in place\n"
        "  2. Run: python scripts/preprocess_dataset.py --dataset <name> --input_dir data/<name>\n"
        "  3. Run: python scripts/prepare_kg.py --dataset <name>\n"
        "  4. Update configs/config.yaml with correct file paths"
    )


if __name__ == "__main__":
    main()
