#!/usr/bin/env bash

set -e
set -x

mypy stanza
ruff check stanza tests scripts
ruff format stanza tests --check
