import os
from flask import Flask, render_template
from flask_uploads import configure_uploads

from webui.setting import config
from webui.blueprints.webui import webui_bp
from webui.extensions import cts


def create_app(config_name=None):
    if config_name is None:
        config_name = os.getenv('FLASK_CONFIG', 'development')

    app = Flask('webui')
    app.config.from_object(config[config_name])

    app.jinja_env.trim_blocks = True
    app.jinja_env.lstrip_blocks = True

    register_extensions(app)
    register_blueprints(app)
    register_errors(app)
    return app


def register_extensions(app):
    # upload config
    configure_uploads(app, cts)


def register_blueprints(app):
    app.register_blueprint(webui_bp)


def register_errors(app):
    # invalid route
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('error.html', error=['Page Not Found', '此頁面不存在']), 404

    # server error
    @app.errorhandler(500)
    def internal_server_error(e):
        return render_template('error.html', error=['Internal Server Error', '伺服器發生錯誤']), 500
