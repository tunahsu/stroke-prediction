from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from webui.extensions import cts

class CTForm(FlaskForm):
    case  = StringField(u'病歷號', validators=[DataRequired()], render_kw={'placeholder': u'病歷號', 'class': 'w-75'})
    ct = FileField(validators=[FileAllowed(cts, u'只能上傳圖片!'), FileRequired(u'請選擇圖片')], render_kw={'style': 'display:none'})
    submit = SubmitField(u'上傳', render_kw={'class': 'btn btn-jelly normal-btn loading'})