"""PACE task implementations.

Importing this module registers the compatible PACE task IDs with mjlab.
"""

from mjlab.utils.lab_api.tasks.importer import import_packages

import_packages(__name__, [".mdp", ".agents"])
