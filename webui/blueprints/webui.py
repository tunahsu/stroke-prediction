import os
import gc
from datetime import datetime
from subprocess import Popen
import multiprocessing as mp
import time
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, jsonify, Response

from webui.extensions import cts
from webui.forms import CTForm

webui_bp = Blueprint('webui', __name__)


def getFiles(path=None):
    files = os.listdir(path)
    files = [list(os.path.splitext(file))+[datetime.fromtimestamp(os.path.getctime(os.path.join(path, file))).strftime("%Y/%m/%d - %H:%M:%S")] for file in files]
    files.sort(key = lambda files: files[2])
    return files


def getAbsPath(path=None):
    basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    path = os.path.join(basedir, path)
    return path



@webui_bp.route('/', methods=('GET', 'POST'))
def index():
    form = CTForm()

    if request.method == 'POST':
        current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        for filename in request.files.getlist('ct'):
            cts.save(filename, folder=current_datetime)

        # Set args
        source = os.path.join(current_app.config['UPLOADED_CTS_DEST'], current_datetime)
        dest = os.path.join(current_app.config['UPLOADED_CTS_DEST'], current_datetime, 'heatmap.gif')
        y_weight = getAbsPath('../checkpoints/yolov5_512_500.pt')
        c_weight = getAbsPath('../checkpoints/3dcnn_d64.h5')

        p = Popen('python ../predict.py -i {} -o {} -yw {} -cw {} -c {}'.format(source, dest, y_weight, c_weight, request.form.get('case')), shell=True)
        p.wait()

        return url_for('static', filename='cts/{}/heatmap.gif'.format(current_datetime))
    return render_template('index.html', form=form, postURL=url_for('webui.index'), filesURL=url_for('webui.files'))



@webui_bp.route('/about', methods=('GET', 'POST'))
def about():
    return render_template('about.html')


@webui_bp.route('/files', methods=('GET', 'POST'))
def files():
    path = current_app.config['UPLOADED_CTS_DEST']
    path = getAbsPath(path)
    folders = sorted(os.listdir(path))
    folders.reverse()

    urls = [url_for('static', filename='cts/{}/heatmap.gif'.format(folder)) for folder in folders]
    cases = []
    prs = []

    for folder in folders:
        txt_path = os.path.join(path, folder, 'result.txt')
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            cases.append(lines[0].strip())
            prs.append(lines[1].strip())
    
    return render_template('files.html', folders=folders, urls=urls, cases=cases, prs=prs)

