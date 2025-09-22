#!/usr/bin/env bash

set -e
set -x 

mypy operator_
ruff check operator_ tests scripts
ruff format operator_ tests --check