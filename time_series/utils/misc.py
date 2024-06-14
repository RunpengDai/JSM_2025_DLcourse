# Copyright (c) Tencent, Inc. and its affiliates. All Rights Reserved.
import errno
import os


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def link_file(src, target):
    """symbol link the source directories to target."""
    if os.path.isdir(target) or os.path.isfile(target):
        os.remove(target)
    os.system('ln -s {} {}'.format(src, target))
