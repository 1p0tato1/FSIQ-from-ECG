# Git Commands — Quick Reference

## Safety Checks (Before Every Commit)
```bash
git status              # No data files (.csv, .pkl, .mat, .xlsx)?
git diff --stat         # Only code/docs changed?
```

## Put your changes online
1. Check in which branch you are
```shell
# make sure you are on thomas_groupA
git branch
# result should show thomas_groupA in green
```
2. Now that you are sure you are NOT in main
```shell
git add . # to add all changes
# or
git add filename # to add a change in a specific file
```
3. Commit your change when you are sure about the change after `git add`.
``` shell
git commit -m "message describing the change"
```
4. To be able to see them online
```shell
git push
```
## Undo a change on stage `add`
```shell
# Unstage a specific file (keep changes in working directory)
git restore --staged filename

# Unstage all files (keep changes)
git restore --staged .

# Alternative (older syntax)
git reset HEAD filename
git reset HEAD .

```
## To get a change from main
```bash
git switch main
git pull origin main
# Now go to your branch
git switch thomas_groupA
git merge main
# you then try to get out of the text editor it appears, you can ask me for help with that
```

## To temporarily save something (to be able to change a branch)
When you want to see other branch and it doesn't let you but you're not sure about your changes:
```bash
# If you want to save work and come back later:
git stash                       # saves everything, clean slate
# ... do other stuff ...
git stash pop
```

## Create another branch
```bash
# Start fresh from main
git switch main
git pull origin main

# Create feature branch
git switch branch_base
git switch -c new_branch

# Work, then commit
git add .
git commit -m "feat: brief description"

# Push branch
git push -u origin new_branch
```

## Common Commands
| Command | Purpose |
|---------|---------|
| `git status` | Show changed files |
| `git diff` | Show unstaged changes |
| `git diff --cached` | Show staged changes |
| `git log --oneline -10` | Recent commits |
| `git branch -a` | List all branches |

## Commit Message Format (Conventional)
```
feat: add new model implementation
fix: correct preprocessing bug
docs: update preprocessing pipeline
refactor: simplify model interface
chore: update dependencies
```