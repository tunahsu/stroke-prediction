import os

basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


class BaseConfig(object):
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev key')
    UPLOADED_CTS_DEST = os.path.join(basedir, 'webui/static/cts')
    UPLOADED_CXRS_DEST = os.path.join(basedir, 'webui/static/cxrs')
  

class DevelopmentConfig(BaseConfig):
    pass


class ProductionConfig(BaseConfig):
    pass


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig
}

