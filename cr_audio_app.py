import os
import re
import sqlite3
import uuid
import datetime
from pathlib import Path
from functools import wraps

from flask import (Flask, request, jsonify, session, redirect,
                   render_template, send_file, abort)

app = Flask(__name__)
app.secret_key = os.environ['SECRET_KEY']

WEB_PASSWORD = os.environ['WEB_PASSWORD']
API_KEY = os.environ['API_KEY']

DATA_DIR = Path('/data')
UPLOADS_DIR = DATA_DIR / 'uploads'
TEMPLATES_DIR = DATA_DIR / 'docx_templates'
RESULTS_DIR = DATA_DIR / 'results'

for d in [UPLOADS_DIR, TEMPLATES_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / 'jobs.db'

CR_TYPE_LABELS = {
    'equipe_educative':        'Équipe éducative',
    'conseil_ecole':           "Conseil d'école",
    'entretien_parent':        'Entretien parent',
    'conseil_maitres':         'Conseil des maîtres',
    'pial_mdph':               'PIAL / MDPH',
    'reunion_rentree':         'Réunion de rentrée',
    'commission_harmonisation': "Commission d'harmonisation",
    'entretien_professionnel': 'Entretien professionnel',
    'reunion_projet':          'Réunion de projet',
    'autre':                   'Autre réunion',
}

VALID_STYLES = ['detaille', 'equilibre', 'humain']


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as db:
        db.execute('''CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            audio_filename TEXT NOT NULL,
            template_filename TEXT NOT NULL,
            output_mode TEXT NOT NULL DEFAULT 'docx',
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL,
            updated_at TEXT,
            log TEXT DEFAULT '',
            error TEXT
        )''')
        for col, defn in [
            ("output_mode",  "TEXT NOT NULL DEFAULT 'docx'"),
            ("log",          "TEXT DEFAULT ''"),
            ("cr_type",      "TEXT NOT NULL DEFAULT 'equipe_educative'"),
            ("cr_style",     "TEXT NOT NULL DEFAULT 'equilibre'"),
        ]:
            try:
                db.execute(f"ALTER TABLE jobs ADD COLUMN {col} {defn}")
            except Exception:
                pass
        db.commit()


init_db()


def student_name(filename):
    """Extrait 'NOM PRENOM' depuis le debut du nom de fichier, ou retourne le label brut (mode TXT)."""
    if not filename.lower().endswith('.docx'):
        return filename
    stem = Path(filename).stem
    m = re.match(
        r'^([A-ZÀÂÄÉÈÊËÎÏÔÙÛÜÇ][A-ZÀÂÄÉÈÊËÎÏÔÙÛÜÇ\- ]+?)'
        r'\s+([A-ZÀÂÄÉÈÊËÎÏÔÙÛÜÇ][A-ZÀÂÄÉÈÊËÎÏÔÙÛÜÇa-zàâäéèêëîïôùûüç]+)',
        stem
    )
    if m:
        return m.group(1) + ' ' + m.group(2)
    return stem


def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if request.headers.get('X-API-Key') != API_KEY:
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated


def require_login(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated


@app.route('/')
@require_login
def index():
    db = get_db()
    jobs = db.execute('SELECT * FROM jobs ORDER BY created_at DESC LIMIT 30').fetchall()
    templates = sorted(f.name for f in TEMPLATES_DIR.glob('*.docx'))
    return render_template('index.html', jobs=jobs, templates=templates,
                           student_name=student_name, cr_type_labels=CR_TYPE_LABELS)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form.get('password') == WEB_PASSWORD:
            session['logged_in'] = True
            return redirect('/')
        return render_template('login.html', error='Mot de passe incorrect')
    return render_template('login.html', error=None)


@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')


@app.route('/upload', methods=['POST'])
@require_login
def upload():
    audio = request.files.get('audio')
    output_mode = request.form.get('output_mode', 'docx')
    if output_mode not in ('docx', 'txt'):
        output_mode = 'docx'

    cr_style = request.form.get('cr_style', 'equilibre')
    if cr_style not in VALID_STYLES:
        cr_style = 'equilibre'

    if not audio:
        return 'Fichier audio requis', 400

    allowed = {'.mp3', '.wav', '.m4a', '.ogg', '.flac', '.mp4', '.mkv', '.aac', '.opus', '.3gp', '.amr'}
    suffix = Path(audio.filename).suffix.lower()
    if suffix not in allowed:
        return 'Format non supporte : ' + suffix, 400

    if output_mode == 'docx':
        template_name = request.form.get('template', '').strip()
        if not template_name or not (TEMPLATES_DIR / template_name).exists():
            return 'Template DOCX introuvable', 400
        cr_type = 'equipe_educative'
    else:
        cr_type = request.form.get('cr_type', 'autre').strip()
        if cr_type not in CR_TYPE_LABELS:
            cr_type = 'autre'
        titre = request.form.get('titre_txt', '').strip()
        template_name = titre if titre else CR_TYPE_LABELS.get(cr_type, 'Compte rendu')

    job_id = str(uuid.uuid4())
    safe_name = job_id + suffix
    (UPLOADS_DIR / safe_name).write_bytes(audio.read())

    now = datetime.datetime.now().isoformat()
    with get_db() as db:
        db.execute(
            'INSERT INTO jobs (id, audio_filename, template_filename, output_mode, cr_type, cr_style, status, created_at, log) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (job_id, safe_name, template_name, output_mode, cr_type, cr_style, 'pending', now, '')
        )
        db.commit()
    return redirect('/')


# ── API worker ─────────────────────────────────────────────────────────────────

@app.route('/api/templates')
@require_api_key
def list_templates():
    return jsonify([f.name for f in TEMPLATES_DIR.glob('*.docx')])


@app.route('/api/templates/<path:name>')
@require_api_key
def get_template(name):
    path = TEMPLATES_DIR / Path(name).name
    if not path.exists():
        abort(404)
    return send_file(str(path))


@app.route('/api/jobs/pending')
@require_api_key
def get_pending_jobs():
    threshold = (datetime.datetime.now() - datetime.timedelta(minutes=45)).isoformat()
    with get_db() as db:
        db.execute(
            "UPDATE jobs SET status='pending' WHERE status='processing' AND updated_at < ?",
            (threshold,)
        )
        db.commit()
    jobs = get_db().execute(
        "SELECT id, audio_filename, template_filename, output_mode, cr_type, cr_style FROM jobs "
        "WHERE status='pending' ORDER BY created_at"
    ).fetchall()
    return jsonify([dict(j) for j in jobs])


@app.route('/api/jobs/<job_id>/audio')
@require_api_key
def get_job_audio(job_id):
    job = get_db().execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
    if not job:
        abort(404)
    audio_path = UPLOADS_DIR / job['audio_filename']
    if not audio_path.exists():
        abort(404)
    now = datetime.datetime.now().isoformat()
    with get_db() as db:
        db.execute("UPDATE jobs SET status='processing', updated_at=? WHERE id=?", (now, job_id))
        db.commit()
    return send_file(str(audio_path))


@app.route('/api/jobs/<job_id>/log', methods=['POST'])
@require_api_key
def append_log(job_id):
    data = request.get_json(silent=True) or {}
    line = str(data.get('message', ''))[:300]
    now = datetime.datetime.now().strftime('%H:%M:%S')
    entry = '[' + now + '] ' + line + '\n'
    with get_db() as db:
        db.execute("UPDATE jobs SET log = COALESCE(log, '') || ? WHERE id=?", (entry, job_id))
        db.commit()
    return jsonify({'status': 'ok'})


@app.route('/api/jobs/<job_id>/complete', methods=['POST'])
@require_api_key
def complete_job(job_id):
    if not get_db().execute('SELECT 1 FROM jobs WHERE id=?', (job_id,)).fetchone():
        abort(404)
    now = datetime.datetime.now().isoformat()
    if 'result' in request.files:
        result = request.files['result']
        ext = request.form.get('ext', 'docx')
        (RESULTS_DIR / (job_id + '_result.' + ext)).write_bytes(result.read())
    with get_db() as db:
        db.execute("UPDATE jobs SET status='completed', updated_at=? WHERE id=?", (now, job_id))
        db.commit()
    return jsonify({'status': 'ok'})


@app.route('/api/jobs/<job_id>/error', methods=['POST'])
@require_api_key
def error_job(job_id):
    data = request.get_json(silent=True) or {}
    error_msg = str(data.get('error', 'Erreur inconnue'))[:500]
    now = datetime.datetime.now().isoformat()
    with get_db() as db:
        db.execute("UPDATE jobs SET status='error', error=?, updated_at=? WHERE id=?",
                   (error_msg, now, job_id))
        db.commit()
    return jsonify({'status': 'ok'})


# ── Logs ───────────────────────────────────────────────────────────────────────

@app.route('/api/logs/poll')
@require_login
def poll_logs():
    jobs = get_db().execute(
        "SELECT id, template_filename, output_mode, cr_type, cr_style, status, log, error "
        "FROM jobs ORDER BY created_at DESC LIMIT 10"
    ).fetchall()
    return jsonify([dict(j) for j in jobs])


@app.route('/logs')
@require_login
def logs_page():
    jobs = get_db().execute(
        "SELECT id, template_filename, output_mode, cr_type, cr_style, status, created_at, updated_at, log, error "
        "FROM jobs ORDER BY created_at DESC LIMIT 10"
    ).fetchall()
    return render_template('logs.html', jobs=jobs, student_name=student_name)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
