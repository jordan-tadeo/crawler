"""Where the laptop's data folder resolves to: the env var, beside the main
checkout (from a worktree too), the old home-folder one, in that order."""

import os

import data_paths as dp


def layout(tmp_path):
    """<folder>/repo with a .git dir and <folder>/worktrees/wt pointing back."""
    folder = tmp_path / "crawler"
    repo = folder / "repo"
    (repo / ".git" / "worktrees" / "wt").mkdir(parents=True)
    (repo / ".git" / "worktrees" / "wt" / "commondir").write_text("../..\n")
    wt = folder / "worktrees" / "wt"
    wt.mkdir(parents=True)
    (wt / ".git").write_text(f"gitdir: {repo / '.git' / 'worktrees' / 'wt'}\n")
    return folder, repo, wt


def test_a_worktree_finds_its_main_checkout(tmp_path):
    folder, repo, wt = layout(tmp_path)
    assert dp.main_checkout(str(repo)) == str(repo)
    assert dp.main_checkout(str(wt)) == str(repo)


def test_data_beside_the_checkout_wins_over_the_old_folder(tmp_path, monkeypatch):
    folder, repo, wt = layout(tmp_path)
    home = tmp_path / "home"
    (home / "crawler-data").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("CRAWLER_DATA", raising=False)
    assert dp.data_root(str(wt)) == str(home / "crawler-data")
    (folder / "data").mkdir()
    assert dp.data_root(str(wt)) == str(folder / "data")
    assert dp.data_root(str(repo)) == str(folder / "data")


def test_the_env_var_wins_and_nothing_found_means_beside(tmp_path, monkeypatch):
    folder, repo, wt = layout(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "nohome"))
    monkeypatch.delenv("CRAWLER_TRAIN", raising=False)
    assert dp.train_dir(str(wt)) == str(folder / "train")
    monkeypatch.setenv("CRAWLER_DATA", str(tmp_path / "elsewhere"))
    assert dp.data_root(str(wt)) == str(tmp_path / "elsewhere")


def test_no_personal_path_in_the_defaults():
    assert all(v.startswith("~/") for v in dp.LEGACY.values())
    assert os.path.isabs(dp.DATA_ROOT)
