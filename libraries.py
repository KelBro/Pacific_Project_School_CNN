from flask import Flask, render_template, request, send_file, session, url_for
from PIL import Image
from datetime import datetime
from functools import lru_cache
from logging.handlers import RotatingFileHandler

import torch
import torchvision.models as models
import os
import logging
import json
