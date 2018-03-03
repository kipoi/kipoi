#!/bin/bash
set -eou pipefail

# References:
#  - https://github.com/bioconda/bioconda-utils/blob/master/.circleci/build-docs.sh
#  - https://docs.travis-ci.com/user/encrypting-files
#  - https://gist.github.com/domenic/ec8b0fc8ab45f39403dd

# ----------------------------------------------------------------------------
#
# Repository-specific configuration
#
# ----------------------------------------------------------------------------

# Note that the keypair needs to be specific to repo, so if ORIGIN changes, the
# keypair (docs/key.enc, and the corresponding public key in the setting of the
# repo) need to be updated.
BRANCH="master"
ORIGIN="kipoi.github.io"
GITHUB_USERNAME="kipoi"
TARGET_FOLDER="docs"

# DOCSOURCE is directory containing the Makefile, relative to the directory
# containing this bash script.
DOCSOURCE=`pwd`/docs

# DOCHTML is where mkdocs is configured to save the output HTML
DOCHTML=$DOCSOURCE/site

# tmpdir to which built docs will be copied
STAGING=/tmp/${GITHUB_USERNAME}-docs

# Build docs only if ci-runner is testing this branch:
BUILD_DOCS_FROM_BRANCH="master"

# TODO - to build the core webpage, modify which folder do get removed

# ----------------------------------------------------------------------------
#
# END repository-specific configuration. The code below is generic; to use for
# another repo, edit the above settings.
#
# ----------------------------------------------------------------------------

if [[ $CIRCLE_PROJECT_USERNAME != kipoi ]]; then
    # exit if not in kipoi repo
    exit 0
fi

REPO="git@github.com:${GITHUB_USERNAME}/${ORIGIN}.git"

# clone the branch to tmpdir, clean out contents
rm -rf $STAGING
mkdir -p $STAGING

SHA=$(git rev-parse --verify HEAD)
git clone $REPO $STAGING
cd $STAGING
git checkout $BRANCH || git checkout --orphan $BRANCH
# remove the existing target folder
rm -rf ${TARGET_FOLDER}

# copy over the docs to tmpdir
cd ${DOCSOURCE}
cp -r ${DOCHTML} $STAGING/${TARGET_FOLDER}

# commit and push
cd $STAGING
touch .nojekyll
git add .nojekyll

# committing with no changes results in exit 1, so check for that case first.
if git diff --quiet; then
    echo "No changes to push -- exiting cleanly"
    exit 0
fi

if [[ $CIRCLE_BRANCH != master ]]; then
    echo "Not pushing docs because not on branch '$BUILD_DOCS_FROM_BRANCH'"
    exit 0
fi


# Add, commit, and push
echo ".*" >> .gitignore
git config user.name "Circle-CI"
git config user.email "${GITHUB_USERNAME}@users.noreply.github.com"
git add -A .
git commit --all -m "Updated docs to commit ${SHA}."
echo "Pushing to $REPO:$BRANCH"
git push $REPO $BRANCH &> /dev/null
