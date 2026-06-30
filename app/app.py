"""Aplicatie Streamlit pentru clasificarea starilor unei imprimante 3D FDM din semnal de vibratie."""

from collections import deque
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import kurtosis, skew

try:
    import serial
    import serial.tools.list_ports
    SERIAL_OK = True
except ImportError:
    SERIAL_OK = False


EXPORT_DIR = Path('export')
WINDOW_SIZE = 512
DECIM_FACTOR = 5
DEFAULT_BAUD = 115200

CULORI = {
    'arm_failure': '#7F77DD', 'bowden': '#1D9E75', 'plastic': '#D85A30',
    'proper': '#D4537E',      'retraction': '#378ADD', 'unstick': '#BA7517',
    'normal': '#1D9E75',      'defect': '#D85A30',
}


def culoare_clasa(label: str) -> str:
    return CULORI.get(label.lower(), CULORI.get(label, '#888888'))


def extract_45(window_3ax: np.ndarray) -> np.ndarray:
    feat = []
    for ax_idx in range(3):
        ax_data = window_3ax[:, ax_idx]
        abs_ax = np.abs(ax_data)
        rms = np.sqrt(np.mean(ax_data ** 2))
        crest = abs_ax.max() / (rms + 1e-9)
        fft_vals = np.abs(np.fft.rfft(ax_data))
        feat += [
            np.mean(ax_data), np.std(ax_data),
            abs_ax.max(), ax_data.min(),
            rms, kurtosis(ax_data),
            crest, skew(ax_data),
            np.mean(fft_vals), np.max(fft_vals),
            float(np.argmax(fft_vals)), np.sum(fft_vals ** 2),
            np.sum(fft_vals[:10]), np.sum(fft_vals[10:50]),
            np.sum(fft_vals[50:]),
        ]
    return np.array(feat, dtype=np.float64)


def _try_load(*paths):
    if not all(Path(p).exists() for p in paths):
        return None
    return [joblib.load(p) for p in paths]


@st.cache_resource
def load_multiclass():
    return _try_load(EXPORT_DIR / 'model_s1.joblib',
                     EXPORT_DIR / 'scaler_s1.joblib',
                     EXPORT_DIR / 'label_encoder.joblib')


@st.cache_resource
def load_binary():
    return _try_load(EXPORT_DIR / 'model_s1_bin.joblib',
                     EXPORT_DIR / 'scaler_s1_bin.joblib',
                     EXPORT_DIR / 'label_encoder_bin.joblib')


@st.cache_resource
def load_replay():
    files = _try_load(EXPORT_DIR / 'model_demo.joblib',
                      EXPORT_DIR / 'scaler.joblib',
                      EXPORT_DIR / 'label_encoder.joblib')
    if files is None or not (EXPORT_DIR / 'demo_set.csv').exists():
        return None
    demo_df = pd.read_csv(EXPORT_DIR / 'demo_set.csv')
    return files + [demo_df]


def available_replay_blocks():
    blocks = []
    for p in EXPORT_DIR.glob('demo_set_b*.csv'):
        try:
            blocks.append(int(p.stem.split('_b')[-1]))
        except ValueError:
            continue
    return sorted(blocks)


@st.cache_resource
def load_replay_block(b: int):
    files = _try_load(EXPORT_DIR / f'model_demo_b{b}.joblib',
                      EXPORT_DIR / f'scaler_demo_b{b}.joblib',
                      EXPORT_DIR / 'label_encoder.joblib')
    if files is None or not (EXPORT_DIR / f'demo_set_b{b}.csv').exists():
        return None
    demo_df = pd.read_csv(EXPORT_DIR / f'demo_set_b{b}.csv')
    return files + [demo_df]


def get_serial_ports() -> list:
    if not SERIAL_OK:
        return []
    return [p.device for p in serial.tools.list_ports.comports()]


def init_live_state():
    defaults = {
        'serial_conn': None,
        'connected': False,
        'buffer_x': deque(maxlen=WINDOW_SIZE),
        'buffer_y': deque(maxlen=WINDOW_SIZE),
        'buffer_z': deque(maxlen=WINDOW_SIZE),
        'last_prediction': None,
        'last_probs': None,
        'pred_history': [],
        'plot_x': deque(maxlen=1000),
        'plot_y': deque(maxlen=1000),
        'plot_z': deque(maxlen=1000),
        'total_samples': 0,
        'decim_counter': 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_live_buffers():
    for k in ['buffer_x', 'buffer_y', 'buffer_z', 'plot_x', 'plot_y', 'plot_z']:
        st.session_state[k].clear()
    st.session_state.pred_history = []
    st.session_state.last_prediction = None
    st.session_state.last_probs = None
    st.session_state.total_samples = 0
    st.session_state.decim_counter = 0


def serial_sidebar():
    if not SERIAL_OK:
        st.sidebar.error('pyserial nu este instalat. Ruleaza: pip install pyserial')
        return False

    ports = get_serial_ports()
    if not ports:
        st.sidebar.warning('Niciun port COM detectat. Conecteaza ESP32 prin USB.')
        if st.sidebar.button('Refresh porturi'):
            st.rerun()
        return False

    port = st.sidebar.selectbox('Port COM', ports, index=0)
    baud = st.sidebar.number_input('Baud rate', value=DEFAULT_BAUD, step=1200)

    if not st.session_state.connected:
        if st.sidebar.button('Conecteaza', use_container_width=True, type='primary'):
            try:
                ser = serial.Serial(port, baud, timeout=0.05)
                time.sleep(2)
                ser.reset_input_buffer()
                st.session_state.serial_conn = ser
                st.session_state.connected = True
                time.sleep(0.3)
                st.rerun()
            except Exception as e:
                st.sidebar.error(f'Eroare conexiune: {e}')
    else:
        if st.sidebar.button('Deconecteaza', use_container_width=True):
            if st.session_state.serial_conn:
                st.session_state.serial_conn.close()
            st.session_state.serial_conn = None
            st.session_state.connected = False
            st.rerun()

    st.sidebar.metric('Sample-uri pastrate', f'{st.session_state.total_samples:,}')
    st.sidebar.metric('Buffer', f'{len(st.session_state.buffer_x)} / {WINDOW_SIZE}')
    if st.sidebar.button('Reset buffere'):
        reset_live_buffers()
        st.rerun()

    return st.session_state.connected


def read_serial_and_predict(model, scaler, encoder):
    ser = st.session_state.serial_conn
    t_start = time.time()
    while time.time() - t_start < 0.2:
        try:
            line = ser.readline().decode(errors='ignore').strip()
            if not line or line == 'READY':
                continue
            parts = line.split(',')
            if len(parts) != 3:
                continue
            ax, ay, az = float(parts[0]), float(parts[1]), float(parts[2])
            if abs(ax) > 200 or abs(ay) > 200 or abs(az) > 200:
                continue
            st.session_state.decim_counter += 1
            if st.session_state.decim_counter % DECIM_FACTOR != 0:
                continue
            st.session_state.buffer_x.append(ax)
            st.session_state.buffer_y.append(ay)
            st.session_state.buffer_z.append(az)
            st.session_state.plot_x.append(ax)
            st.session_state.plot_y.append(ay)
            st.session_state.plot_z.append(az)
            st.session_state.total_samples += 1
        except (ValueError, serial.SerialException):
            continue

    if len(st.session_state.buffer_x) >= WINDOW_SIZE:
        arr = np.column_stack([np.array(st.session_state.buffer_x),
                               np.array(st.session_state.buffer_y),
                               np.array(st.session_state.buffer_z)])
        features = extract_45(arr).reshape(1, -1)
        probs = model.predict_proba(scaler.transform(features))[0]
        pred_idx = int(np.argmax(probs))
        pred_label = encoder.inverse_transform([pred_idx])[0]
        st.session_state.last_prediction = pred_label
        st.session_state.last_probs = probs
        st.session_state.pred_history.append(
            (time.strftime('%H:%M:%S'), pred_label, float(probs[pred_idx])))
        if len(st.session_state.pred_history) > 30:
            st.session_state.pred_history = st.session_state.pred_history[-30:]
        st.session_state.buffer_x.clear()
        st.session_state.buffer_y.clear()
        st.session_state.buffer_z.clear()


def render_prediction(encoder, big: bool = False):
    col1, col2 = st.columns([1, 1.3])
    with col1:
        st.subheader('Stare curenta')
        if st.session_state.last_prediction:
            pred = st.session_state.last_prediction
            conf = st.session_state.last_probs[int(np.argmax(st.session_state.last_probs))]
            font = '52px' if big else '38px'
            st.markdown(
                f"""
                <div style="background:{culoare_clasa(pred)}; padding:36px;
                            border-radius:12px; text-align:center; color:white;">
                    <div style="font-size:14px; opacity:0.85;">PREDICTIE</div>
                    <div style="font-size:{font}; font-weight:800; margin:6px 0;">
                        {pred.upper()}</div>
                    <div style="font-size:16px;">Confidence: {conf*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info(f'Se acumuleaza sample-uri ({len(st.session_state.buffer_x)} / {WINDOW_SIZE}).')

    with col2:
        st.subheader('Probabilitati')
        if st.session_state.last_probs is not None:
            pdf = pd.DataFrame({'Clasa': encoder.classes_,
                                'Prob (%)': st.session_state.last_probs * 100}
                               ).sort_values('Prob (%)', ascending=True)
            fig = go.Figure(go.Bar(
                x=pdf['Prob (%)'], y=pdf['Clasa'], orientation='h',
                marker_color=[culoare_clasa(c) for c in pdf['Clasa']],
                text=[f'{v:.1f}%' for v in pdf['Prob (%)']], textposition='outside'))
            fig.update_layout(xaxis_range=[0, 110], height=280,
                              margin=dict(l=0, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    if len(st.session_state.plot_x) > 50:
        st.markdown('---')
        st.subheader('Vibratii in timp real (dupa decimare)')
        x_axis = np.arange(len(st.session_state.plot_x))
        fig = go.Figure()
        for buf, nume in [(st.session_state.plot_x, 'acc_x'),
                          (st.session_state.plot_y, 'acc_y'),
                          (st.session_state.plot_z, 'acc_z')]:
            fig.add_trace(go.Scatter(x=x_axis, y=list(buf), name=nume, line=dict(width=1)))
        fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=10),
                          xaxis_title='Sample', yaxis_title='Acc (m/s^2)')
        st.plotly_chart(fig, use_container_width=True)


def mode_live(task: str):
    init_live_state()
    artifacts = load_multiclass() if task == 'multi' else load_binary()
    if artifacts is None:
        need = ('model_s1' if task == 'multi' else 'model_s1_bin')
        st.error(f'Lipsesc fisierele pentru acest mod ({need}.joblib etc.) din "{EXPORT_DIR}/".')
        return
    model, scaler, encoder = artifacts

    titlu = 'Live - clasificare in 6 stari' if task == 'multi' else 'Live - Normal vs. Defect'
    st.title(titlu)
    st.caption('Model pe un singur senzor (3 axe, 45 caracteristici). '
               f'Date decimate la ~200 Hz, ferestre de {WINDOW_SIZE} esantioane.')

    st.sidebar.header('Conexiune ESP32')
    connected = serial_sidebar()

    if connected and st.session_state.serial_conn:
        read_serial_and_predict(model, scaler, encoder)

    render_prediction(encoder, big=(task == 'binary'))

    if st.session_state.pred_history:
        st.markdown('---')
        st.subheader('Istoric predictii')
        hist = pd.DataFrame(st.session_state.pred_history[-20:],
                            columns=['Ora', 'Predictie', 'Confidence'])
        hist['Confidence'] = hist['Confidence'].map(lambda x: f'{x*100:.1f}%')
        st.dataframe(hist.iloc[::-1], use_container_width=True, hide_index=True)

    if st.session_state.connected:
        time.sleep(0.05)
        st.rerun()


def mode_replay():
    st.title('Replay pe date din set (fara hardware)')

    blocks = available_replay_blocks()
    if blocks:
        b = st.sidebar.selectbox('Bloc holdout', blocks, index=len(blocks) - 1)
        if st.session_state.get('rp_block') != b:
            st.session_state.rp_block = b
            st.session_state.rp_idx = 0
            st.session_state.rp_running = False
            st.session_state.rp_hist = []
        artifacts = load_replay_block(b)
        eticheta_bloc = f'blocul {b}'
    else:
        artifacts = load_replay()
        eticheta_bloc = 'blocul 9'

    if artifacts is None:
        st.error('Lipsesc fisierele pentru replay din "export/": model_demo*, scaler*, demo_set*.csv.')
        return
    model, scaler, encoder, demo_df = artifacts
    feature_cols = [c for c in demo_df.columns if c not in ('label_true', 'label_idx')]

    st.caption(f'Model testat pe {eticheta_bloc}: {len(demo_df)} ferestre care nu au fost '
               'folosite la antrenare.')

    if 'rp_idx' not in st.session_state:
        st.session_state.rp_idx = 0
        st.session_state.rp_running = False
        st.session_state.rp_hist = []

    speed = st.sidebar.slider('Viteza (ferestre/s)', 0.5, 5.0, 2.0, 0.5)
    c1, c2 = st.sidebar.columns(2)
    with c1:
        if not st.session_state.rp_running:
            if st.button('Start', use_container_width=True, type='primary'):
                st.session_state.rp_running = True
                st.rerun()
        else:
            if st.button('Pauza', use_container_width=True):
                st.session_state.rp_running = False
                st.rerun()
    with c2:
        if st.button('Reset', use_container_width=True):
            st.session_state.rp_idx = 0
            st.session_state.rp_running = False
            st.session_state.rp_hist = []
            st.rerun()

    st.sidebar.metric('Fereastra', f'{st.session_state.rp_idx} / {len(demo_df)}')
    if st.session_state.rp_hist:
        ok = sum(1 for _, _, _, c in st.session_state.rp_hist if c)
        st.sidebar.metric('Acuratete rolling',
                          f'{ok/len(st.session_state.rp_hist)*100:.1f}%')

    if st.session_state.rp_idx >= len(demo_df):
        st.success('Replay terminat. Apasa Reset pentru a relua.')
        st.session_state.rp_running = False
        return

    row = demo_df.iloc[st.session_state.rp_idx]
    feats = row[feature_cols].values.reshape(1, -1).astype(np.float64)
    probs = model.predict_proba(scaler.transform(feats))[0]
    pred_idx = int(np.argmax(probs))
    pred = encoder.inverse_transform([pred_idx])[0]
    true = row['label_true']
    ok = (pred == true)
    st.session_state.rp_hist.append((st.session_state.rp_idx, pred, true, ok))
    if len(st.session_state.rp_hist) > 30:
        st.session_state.rp_hist = st.session_state.rp_hist[-30:]

    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.subheader('Predictie')
        st.markdown(
            f"""<div style="background:{culoare_clasa(pred)}; padding:30px;
                    border-radius:12px; text-align:center; color:white;">
                <div style="font-size:14px; opacity:.85;">PREDICTIE</div>
                <div style="font-size:38px; font-weight:800; margin:6px 0;">{pred.upper()}</div>
                <div style="font-size:16px;">Confidence: {probs[pred_idx]*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)
        verdict = 'CORECT' if ok else 'GRESIT'
        st.markdown(
            f"""<div style="border:2px solid {culoare_clasa(true)}; margin-top:12px;
                    padding:12px; border-radius:8px; text-align:center;">
                <div style="font-size:13px; color:#666;">ETICHETA REALA</div>
                <div style="font-size:22px; font-weight:700;">{true}</div>
                <div style="margin-top:4px;">{verdict}</div>
                </div>""", unsafe_allow_html=True)
    with c2:
        st.subheader('Probabilitati')
        pdf = pd.DataFrame({'Clasa': encoder.classes_, 'Prob (%)': probs * 100}
                           ).sort_values('Prob (%)', ascending=True)
        fig = go.Figure(go.Bar(x=pdf['Prob (%)'], y=pdf['Clasa'], orientation='h',
                               marker_color=[culoare_clasa(c) for c in pdf['Clasa']],
                               text=[f'{v:.1f}%' for v in pdf['Prob (%)']],
                               textposition='outside'))
        fig.update_layout(xaxis_range=[0, 110], height=300,
                          margin=dict(l=0, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    if st.session_state.rp_running:
        time.sleep(1.0 / speed)
        st.session_state.rp_idx += 1
        st.rerun()


def mode_csv():
    st.title('Analiza fisier CSV colectat')
    st.caption('Incarca un CSV (timestamp, acc_x, acc_y, acc_z, label) si afiseaza '
               f'predictiile pe fiecare fereastra de {WINDOW_SIZE} esantioane.')

    tip_analiza = st.radio('Model folosit pentru analiza',
                           ['6 clase', 'Normal/Defect'], horizontal=True)

    artifacts = load_multiclass() if tip_analiza == '6 clase' else load_binary()
    if artifacts is None:
        if tip_analiza == '6 clase':
            st.error('Lipseste model_s1.joblib / scaler_s1.joblib / label_encoder.joblib din "export/".')
        else:
            st.error('Lipseste model_s1_bin.joblib / scaler_s1_bin.joblib / label_encoder_bin.joblib din "export/".')
        return
    model, scaler, encoder = artifacts

    ultim = st.session_state.get('ultimul_csv')
    sursa = None
    if ultim and Path(ultim).exists():
        if st.button(f'Foloseste ultimul fisier colectat ({ultim})'):
            sursa = ultim

    up = st.file_uploader('Sau alege un fisier CSV', type=['csv', 'txt'])
    if up is not None:
        sursa = up

    if sursa is None:
        st.info('Incarca un CSV sau colecteaza unul din modul "Colectare date".')
        return

    df = pd.read_csv(sursa)
    if {'acc_x', 'acc_y', 'acc_z'}.issubset(df.columns):
        data = df[['acc_x', 'acc_y', 'acc_z']].values.astype(np.float64)
    elif {'acc1_x', 'acc1_y', 'acc1_z'}.issubset(df.columns):
        data = df[['acc1_x', 'acc1_y', 'acc1_z']].values.astype(np.float64)
    else:
        st.error('Nu gasesc coloanele de acceleratie (acc_x/acc_y/acc_z).')
        return

    mask = ~np.isnan(data).any(axis=1) & (np.abs(data).max(axis=1) < 100)
    data = data[mask]
    label_true = df['label'].iloc[0] if 'label' in df.columns else None

    n_win = (len(data) - WINDOW_SIZE) // WINDOW_SIZE
    if n_win <= 0:
        st.error(f'Prea putine sample-uri valide ({len(data)}) pentru o fereastra de {WINDOW_SIZE}.')
        return

    feats = np.array([extract_45(data[i*WINDOW_SIZE:(i+1)*WINDOW_SIZE]) for i in range(n_win)])
    probs = model.predict_proba(scaler.transform(feats))
    preds = encoder.inverse_transform(np.argmax(probs, axis=1))

    counts = pd.Series(preds).value_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric('Sample-uri valide', f'{len(data):,}')
    c2.metric('Ferestre analizate', n_win)
    c3.metric('Verdict majoritar', counts.index[0])

    st.subheader('Distributia predictiilor')
    dist = pd.DataFrame({'Clasa': counts.index, 'Ferestre': counts.values})
    fig = go.Figure(go.Bar(x=dist['Clasa'], y=dist['Ferestre'],
                           marker_color=[culoare_clasa(c) for c in dist['Clasa']],
                           text=dist['Ferestre'], textposition='outside'))
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=10),
                      yaxis_title='Nr. ferestre')
    st.plotly_chart(fig, use_container_width=True)

    if label_true is not None:
        label_eval = label_true
        if tip_analiza == 'Normal/Defect':
            label_lower = str(label_true).lower()
            clase_defect = {'arm_failure', 'bowden', 'plastic', 'retraction', 'unstick', 'defect'}
            if 'proper' in label_lower or label_lower == 'normal':
                label_eval = 'normal'
            elif label_lower in clase_defect:
                label_eval = 'defect'
            else:
                label_eval = None

        if label_eval is not None and label_eval in set(encoder.classes_):
            match = (preds == label_eval).mean() * 100
            st.info(f'Eticheta din CSV: **{label_true}** (evaluata ca **{label_eval}**). '
                    f'Concordanta cu predictia: **{match:.1f}%** din ferestre.')
        else:
            st.info(f'Eticheta din CSV: **{label_true}**. Concordanta nu se calculeaza automat, '
                    'deoarece eticheta nu corespunde claselor modelului selectat.')

    st.subheader('Predictii pe primele ferestre')
    tabel = pd.DataFrame({
        'Fereastra': np.arange(min(30, n_win)),
        'Predictie': preds[:30],
        'Confidence': [f'{probs[i].max()*100:.1f}%' for i in range(min(30, n_win))],
    })
    st.dataframe(tabel, use_container_width=True, hide_index=True)


def mode_collect():
    st.title('Colectare date de la senzor (ESP32 + MPU-6050)')
    st.caption('Inregistreaza un semnal etichetat si salveaza un CSV '
               '(timestamp, acc_x, acc_y, acc_z, label).')

    if not SERIAL_OK:
        st.error('pyserial nu este instalat. Ruleaza: pip install pyserial')
        return

    ports = get_serial_ports()
    if not ports:
        st.warning('Niciun port COM detectat. Conecteaza ESP32 prin USB.')
        if st.button('Refresh porturi'):
            st.rerun()
        return

    c1, c2 = st.columns(2)
    with c1:
        port = st.selectbox('Port COM', ports, index=0)
        eticheta = st.text_input('Eticheta (clasa)', value='proper')
    with c2:
        baud = st.number_input('Baud rate', value=DEFAULT_BAUD, step=1200)
        durata = st.number_input('Durata (secunde)', value=30, min_value=5, step=5)

    skip_primele = 100
    valoare_max = 100

    if st.button('Start colectare', type='primary'):
        try:
            ser = serial.Serial(port, baud, timeout=1)
        except Exception as e:
            st.error(f'Nu pot deschide {port}: {e}. '
                     'Inchide Arduino Serial Monitor daca este deschis.')
            return

        time.sleep(2)
        ser.reset_input_buffer()

        progres = st.progress(0.0)
        status = st.empty()
        rows = []
        nr_total = 0
        nr_outlier = 0
        t0 = time.time()

        while time.time() - t0 < durata:
            try:
                line = ser.readline().decode(errors='ignore').strip()
            except serial.SerialException:
                break
            if not line or line == 'READY':
                continue
            parts = line.split(',')
            if len(parts) != 3:
                continue
            try:
                ax, ay, az = float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError:
                continue

            nr_total += 1
            if nr_total <= skip_primele:
                continue
            if abs(ax) > valoare_max or abs(ay) > valoare_max or abs(az) > valoare_max:
                nr_outlier += 1
                continue

            t = time.time() - t0
            rows.append([round(t, 4), round(ax, 4), round(ay, 4), round(az, 4), eticheta])

            elapsed = time.time() - t0
            progres.progress(min(elapsed / durata, 1.0))
            if len(rows) % 200 == 0:
                rate = len(rows) / elapsed if elapsed > 0 else 0
                status.write(f'{elapsed:4.1f}s - {len(rows)} sample-uri salvate - ~{rate:.0f} Hz')

        ser.close()
        progres.progress(1.0)

        if len(rows) < WINDOW_SIZE:
            st.error(f'Doar {len(rows)} sample-uri valide (sub {WINDOW_SIZE}). '
                     'Colecteaza mai mult sau verifica senzorul.')
            return

        df = pd.DataFrame(rows, columns=['timestamp', 'acc_x', 'acc_y', 'acc_z', 'label'])
        nume = f'date_{eticheta}_{int(time.time())}.csv'
        df.to_csv(nume, index=False)
        st.session_state['ultimul_csv'] = nume

        durata_reala = time.time() - t0
        st.success(f'Colectare finalizata: {len(df)} sample-uri in {durata_reala:.1f}s '
                   f'(~{len(df)/durata_reala:.0f} Hz). Salvat ca {nume}.')
        st.caption(f'Outlieri ignorati: {nr_outlier}. '
                   f'Ferestre posibile de {WINDOW_SIZE}: {len(df)//WINDOW_SIZE}.')

        st.download_button('Descarca CSV', data=df.to_csv(index=False),
                           file_name=nume, mime='text/csv')
        st.info('Treci la modul "Analiza fisier CSV" si incarca fisierul salvat.')


def main():
    st.set_page_config(page_title='Detectie defecte imprimante 3D', layout='wide')

    st.sidebar.title('Detectie defecte imprimante 3D')
    mod = st.sidebar.radio(
        'Mod de lucru',
        ['Live - 6 clase', 'Live - Normal/Defect', 'Colectare date',
         'Analiza fisier CSV', 'Replay dataset'],
    )
    st.sidebar.markdown('---')

    if mod == 'Live - 6 clase':
        mode_live('multi')
    elif mod == 'Live - Normal/Defect':
        mode_live('binary')
    elif mod == 'Colectare date':
        mode_collect()
    elif mod == 'Analiza fisier CSV':
        mode_csv()
    else:
        mode_replay()


if __name__ == '__main__':
    main()
