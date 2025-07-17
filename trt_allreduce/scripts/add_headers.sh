#!/bin/bash

# This script adds a standardized file header to all source and header files
# under src/ and include/ directories, if they don't already contain it.

HEADER="/**
 * @file %FILENAME%
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief [DESCRIPTION HERE]
 * @version 0.1
 * @date 2025-07-17
 *
 * @copyright Copyright (c) 2025
 *
 */"

find ../src ../include \
  -type f \( -name "*.c*" -o -name "*.h*" \) | while read -r file; do
  if ! grep -q "@file" "$file"; then
    filename=$(basename "$file")
    header=$(echo "$HEADER" | sed "s/%FILENAME%/$filename/")
    # Insert header at the top of the file
    printf "%s" "$header" | cat - "$file" > "/tmp/header.$$" && mv "/tmp/header.$$" "$file"
    echo "Added header to $file"
  fi
done
