#!/usr/bin/env bash

set -eux

pixi run postinstall
pixi run pre-commit-install
pixi run generate-opset
pixi run pre-commit-run
