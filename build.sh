#!/bin/bash
set -e

# The Render free instance has a 512 MB memory limit. The app uses the
# lightweight rules-only masker by default, so there is no model download here.
