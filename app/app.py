import streamlit as st
from openai import OpenAI
import os
import re
import json
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── Import DuckDB (opcional) ──
try:
    import duckdb
    DUCKDB_DISPONIBLE = True
except ImportError:
    DUCKDB_DISPONIBLE = False
    duckdb = None

# ── Import adlfs (opcional) ──
try:
    import adlfs
    ADLFS_DISPONIBLE = True
except ImportError:
    ADLFS_DISPONIBLE = False
    adlfs = None

# ── Import pyodbc (opcional) ──
try:
    import pyodbc
    PYODBC_DISPONIBLE = True
except ImportError:
    PYODBC_DISPONIBLE = False
    pyodbc = None

st.set_page_config(page_title="Sistema UCI - Portal de Información", layout="wide", page_icon="⚕️")

# ==========================================
# 1. CONFIGURACIÓN DE PROVEEDORES DE IA
# ==========================================
PROVIDERS_CONFIG = {
    "REMOTE_OLLAMA": {"base_url": "https://proyectofindemaster.nasmourenza.es/v1", "api_key": "ollama", "model": "Mistral-nemo:latest"},
    "GROQ": {"base_url": "https://api.groq.com/openai/v1", "api_key": os.getenv("GROQ_API_KEY", ""), "model": "llama-3.1-8b-instant"},
    "OPENCODE_GO": {"base_url": "https://opencode.ai/zen/go/v1", "api_key": os.getenv("OPENCODE_API_KEY", ""), "model": "deepseek-v4-flash"},
}

with st.sidebar:
    st.header("⚙️ Configuración del Sistema")

    default_provider = os.getenv("LLM_PROVIDER", "OPENCODE_GO")
    provider_keys = list(PROVIDERS_CONFIG.keys())
    if default_provider not in provider_keys:
        default_provider = "OPENCODE_GO"

    backend_seleccionado = st.selectbox(
        "Proveedor de Inteligencia Artificial",
        options=provider_keys,
        index=provider_keys.index(default_provider)
    )

    st.markdown("---")
    st.header("🔗 Origen de Datos Clínicos")

    modos_datos = ["Text-to-SQL + ADLS (DuckDB)", "Synapse SQL Pool (pyodbc)"]
    modo_datos = st.selectbox("Motor de consulta", options=modos_datos, index=0)

    if modo_datos == "Synapse SQL Pool (pyodbc)" and not PYODBC_DISPONIBLE:
        st.warning("⚠️ pyodbc no disponible. Instálalo o usa el modo DuckDB.")
    if modo_datos == "Text-to-SQL + ADLS (DuckDB)" and not DUCKDB_DISPONIBLE:
        st.warning("⚠️ duckdb no disponible. Instálalo con: pip install duckdb")

    st.markdown("---")
    st.header("🔐 Control de Acceso")
    codigo_acceso = st.text_input("Introduce tu Código de Acceso", type="password")

cfg = PROVIDERS_CONFIG[backend_seleccionado]
client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
MODELO_LLM = cfg["model"]

# ==========================================
# 2. CONTROL DE ACCESO (RBAC por Código)
# ==========================================
rol_usuario = None
if codigo_acceso == "007":
    rol_usuario = "Personal Médico"
elif codigo_acceso == "000":
    rol_usuario = "Familiar"


# ==========================================
# 3.1 BACKEND: TEXT-TO-SQL + ADLS (DUCKDB)
# ==========================================
STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT", "stproyectomastergrupo3")
BASE_SILVER = f"abfss://medicalproyect-processed@{STORAGE_ACCOUNT}.dfs.core.windows.net/silver"
BASE_GOLD = f"abfss://medicalproyect-curated@{STORAGE_ACCOUNT}.dfs.core.windows.net/gold/machine_learning"

AZURE_STORAGE_KEY = os.getenv("AZURE_STORAGE_KEY", "")


def _obtener_fs():
    """Crea filesystem de ADLS usando clave de acceso o DefaultAzureCredential."""
    if AZURE_STORAGE_KEY:
        st.caption(f"🔑 Conectando a {STORAGE_ACCOUNT} con clave de acceso...")
        return adlfs.AzureBlobFileSystem(
            account_name=STORAGE_ACCOUNT,
            account_key=AZURE_STORAGE_KEY
        )
    else:
        st.caption("🔑 No hay clave de acceso. Intentando DefaultAzureCredential...")
        from azure.identity import DefaultAzureCredential
        return adlfs.AzureBlobFileSystem(
            account_name=STORAGE_ACCOUNT,
            credential=DefaultAzureCredential()
        )


def _leer_tabla_adls(fs, tabla):
    """Lee una tabla Silver completa desde ADLS usando adlfs."""
    paths = fs.glob(f"medicalproyect-processed/silver/{tabla}/*.parquet")
    real_files = [p for p in paths if "part-" in p and p.endswith(".parquet")]
    if not real_files:
        return None
    dfs = [pd.read_parquet(fs.open(p)) for p in real_files]
    return pd.concat(dfs, ignore_index=True)


def consultar_paciente_duckdb(patient_id):
    """Busca paciente en Silver+Gold usando adlfs + pandas."""
    try:
        fs = _obtener_fs()

        # Cargar patients completo y filtrar
        df_pats = _leer_tabla_adls(fs, "patients")
        if df_pats is None:
            st.warning("No se pudo leer la tabla de pacientes")
            return None

        row = df_pats[df_pats["patient_id"] == patient_id]
        if len(row) == 0:
            return None
        row = row.iloc[0]

        diagnosticos = []
        for dx in ["dx_hypertension", "dx_type2_diabetes", "dx_heart_failure",
                    "dx_chronic_kidney_disease", "dx_copd"]:
            if dx in row.index and row[dx]:
                diagnosticos.append(dx.replace("dx_", "").replace("_", " ").title())

        prob_reingreso = 0.0
        try:
            pred_paths = fs.glob("medicalproyect-curated/gold/machine_learning/icu_predictions/*.parquet")
            real_pred = [p for p in pred_paths if "part-" in p and p.endswith(".parquet")]
            if real_pred:
                df_preds = pd.read_parquet(fs.open(real_pred[0]))
                match = df_preds[df_preds["patient_id"] == patient_id]["probability"]
                if len(match) > 0:
                    prob_reingreso = float(match.iloc[0])
        except Exception:
            pass

        estancia = None
        try:
            df_outs = _leer_tabla_adls(fs, "outcomes")
            if df_outs is not None:
                match = df_outs[df_outs["patient_id"] == patient_id]["length_of_stay_days"]
                if len(match) > 0:
                    estancia = int(match.iloc[0])
        except Exception:
            pass

        # Si no hay predicción del modelo, calcular riesgo base por edad y Charlson
        if prob_reingreso == 0.0:
            edad_val = int(row["age"]) if pd.notna(row["age"]) else 50
            charlson_val = int(row["charlson_index"]) if pd.notna(row["charlson_index"]) else 0
            prob_reingreso = min(0.12 + charlson_val * 0.04 + max(0, edad_val - 55) * 0.003, 0.85)
            prob_reingreso = round(prob_reingreso, 2)

        return {
            "patient_id": row["patient_id"],
            "edad": int(row["age"]) if pd.notna(row["age"]) else "?",
            "genero": row["sex"] if pd.notna(row["sex"]) else "?",
            "dias_estancia": estancia or "?",
            "diagnosticos": ", ".join(diagnosticos) if diagnosticos else "Sin diagnósticos registrados",
            "probabilidad_reingreso": prob_reingreso,
            "charlson": int(row["charlson_index"]) if pd.notna(row["charlson_index"]) else 0,
        }
    except Exception as e:
        st.warning(f"Falta de conexion con la base de datos, entrando en modo demo")
        st.error(f"Detalle: {type(e).__name__}: {str(e)[:200]}")
        return None


# ==========================================
# 3.2 BACKEND: SYNAPSE SQL POOL (PYODBC)
# ==========================================
def consultar_paciente_sqlpool(patient_id):
    """Busca paciente en Synapse SQL Pool usando pyodbc (versión original del compañero)."""
    server = os.getenv("SYNAPSE_SERVER")
    database = os.getenv("SYNAPSE_DB")
    username = os.getenv("SYNAPSE_USER")
    password = os.getenv("SYNAPSE_PASSWORD")

    if not all([server, database, username, password]):
        st.warning("⚠️ Variables de entorno de Synapse no configuradas.")
        return None

    conn_str = f"Driver={{ODBC Driver 18 for SQL Server}};Server={server},1433;Database={database};Uid={username};Pwd={password};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=15;"

    try:
        with pyodbc.connect(conn_str) as conn:
            with conn.cursor() as cursor:
                query = """
                    SELECT
                        p.patient_id, p.edad, p.genero,
                        i.dias_estancia, i.estado_general,
                        d.diagnosticos_concatenados,
                        pr.probabilidad_reingreso
                    FROM dim_paciente p
                    LEFT JOIN fact_ingresos i ON p.patient_id = i.patient_id
                    LEFT JOIN fact_diagnosticos d ON p.patient_id = d.patient_id
                    LEFT JOIN fact_predicciones_ml pr ON p.patient_id = pr.patient_id
                    WHERE p.patient_id = ?
                """
                cursor.execute(query, (patient_id,))
                row = cursor.fetchone()

                if row:
                    return {
                        "patient_id": row[0],
                        "edad": row[1],
                        "genero": row[2],
                        "dias_estancia": row[3],
                        "estado_general": row[4],
                        "diagnosticos": row[5] if row[5] else "Sin diagnósticos registrados",
                        "probabilidad_reingreso": float(row[6]) if row[6] else 0.0,
                        "charlson": 0,
                    }
        return None
    except Exception as e:
        st.error(f"Error con Synapse SQL Pool: {e}")
        return None


# ==========================================
# 3.3 CONSULTA CENTRALIZADA DE PACIENTE
# ==========================================
def consultar_datos_paciente(patient_id):
    """Usa el backend seleccionado para consultar datos del paciente."""
    if modo_datos == "Synapse SQL Pool (pyodbc)":
        resultado = consultar_paciente_sqlpool(patient_id)
    elif modo_datos == "Text-to-SQL + ADLS (DuckDB)":
        resultado = consultar_paciente_duckdb(patient_id)
    else:
        resultado = None

    # Fallback a demo si no hay resultado
    if resultado is None and patient_id == "P0003122":
        return {
            "patient_id": "P0003122", "edad": 68, "genero": "M", "dias_estancia": 6,
            "diagnosticos": "Hypertension, Chronic Kidney Disease",
            "probabilidad_reingreso": 0.74, "charlson": 3, "estado_general": "Estable"
        }

    return resultado


# ==========================================
# 4. SYSTEM PROMPT PARA TEXT-TO-SQL
# ==========================================
SYSTEM_PROMPT_SQL = """Eres un asistente que genera consultas SQL sobre un dataset hospitalario.
Tienes disponibles estas tablas en DuckDB con datos de Silver:

### Tabla: patients (datos demograficos y comorbilidades)
Columnas: patient_id, age, sex, bmi, systolic_bp, diastolic_bp, heart_rate, temperature_c,
smoking_status, alcohol_use, exercise_level, insurance_type, charlson_index,
dx_hypertension, dx_type2_diabetes, dx_hyperlipidemia, dx_obesity,
dx_coronary_artery_disease, dx_heart_failure, dx_atrial_fibrillation,
dx_chronic_kidney_disease, dx_copd, dx_asthma, dx_depression, dx_anxiety,
dx_hypothyroidism, dx_osteoarthritis, dx_type1_diabetes
IMPORTANTE: Las columnas dx_* (comorbilidades como diabetes, hipertension) SOLO estan en esta tabla "patients".
NO uses diagnoses.dx_* ni d.dx_* porque la tabla diagnoses NO tiene columnas dx_*.

### Tabla: outcomes (hospitalizaciones)
Columnas: patient_id, admission_date, discharge_date, length_of_stay_days,
icu_admission, icu_days, in_hospital_death, discharge_disposition,
readmitted_30d, days_to_readmission, primary_drg, total_charges_usd
NOTA: NO existe la columna "risk_of_readmission", "risk_score", "readmission_risk" ni similar.
Para reingreso usa SOLO readmitted_30d.

### Tabla: lab_results (analiticas de laboratorio)
Columnas: patient_id, test_date, test_name, value, unit, reference_low,
reference_high, flag, is_abnormal, delta_from_normal

### Tabla: medications (medicacion recetada)
Columnas: patient_id, medication, dose, unit, frequency, indication,
start_date, duration_days, is_generic, adherence_pct

### Tabla: diagnoses (diagnosticos por visita)
Columnas: patient_id, visit_date, visit_type, primary_diagnosis,
primary_icd10, secondary_diagnoses, secondary_icd10s, provider_specialty
ATENCION: Esta tabla NO tiene columnas dx_*. NUNCA hagas diagnoses.dx_hypertension ni d.dx_hypertension.
Para comorbilidades SIEMPRE usa patients.dx_* (ej: patients.dx_type2_diabetes).

### EJEMPLOS:
Pregunta: "Pacientes mas mayores con diabetes tipo 2"
SQL: SELECT patient_id, age, sex, charlson_index FROM patients WHERE dx_type2_diabetes = TRUE ORDER BY age DESC LIMIT 10

Pregunta: "Analiticas del paciente P0003122"
SQL: SELECT test_date, test_name, value, unit, flag FROM lab_results WHERE patient_id = 'P0003122' ORDER BY test_date LIMIT 50

Pregunta: "Pacientes con diabetes readmitidos en 30 dias"
SQL: SELECT p.patient_id, p.age, p.sex FROM patients p JOIN outcomes o ON p.patient_id = o.patient_id WHERE p.dx_type2_diabetes = TRUE AND o.readmitted_30d = TRUE LIMIT 20

Pregunta: "Coste medio por diagnostico principal"
SQL: SELECT d.primary_diagnosis, COUNT(*) as num, ROUND(AVG(o.total_charges_usd), 2) as coste_medio FROM outcomes o JOIN diagnoses d ON o.patient_id = d.patient_id GROUP BY d.primary_diagnosis ORDER BY coste_medio DESC LIMIT 10

Las rutas base son:
- Silver: abfss://medicalproyect-processed@stproyectomastergrupo3.dfs.core.windows.net/silver/

Usa la sintaxis: SELECT ... FROM read_parquet('ruta/TABLA/*.parquet')  o simplemente el nombre de la tabla.
IMPORTANTE: NO inventes columnas que no aparecen en el listado. Si no está listado, no existe.
LIMITA resultados a 50 filas.
Devuelve SOLO el SQL, sin explicaciones."""


def generar_sql(pregunta, patient_id=None):
    contexto = ""
    if patient_id:
        contexto = f"\nContexto: el usuario pregunta sobre el paciente {patient_id}. Filtra por ese patient_id cuando corresponda.\n"

    try:
        resp = client.chat.completions.create(
            model=MODELO_LLM,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_SQL + contexto},
                {"role": "user", "content": pregunta}
            ],
            temperature=0.1,
            max_tokens=500
        )
        sql = resp.choices[0].message.content.strip()
        sql = re.sub(r"```sql|```", "", sql).strip()
        return sql
    except Exception as e:
        st.error(f"Error generando SQL: {e}")
        return None


def ejecutar_sql(sql):
    try:
        con = duckdb.connect()
        fs = _obtener_fs()
        for tabla in ["patients", "outcomes", "medications", "lab_results", "diagnoses"]:
            df = _leer_tabla_adls(fs, tabla)
            if df is not None:
                con.register(tabla, df)

        resultado = con.execute(sql)
        cols = [desc[0] for desc in resultado.description]
        filas = resultado.fetchall()
        return pd.DataFrame(filas, columns=cols)
    except Exception as e:
        st.error(f"Error ejecutando SQL: {e}")
        return None


# ==========================================
# 5. FLUJO DE EJECUCIÓN E INTERFAZ SEGÚN ROL
# ==========================================
if rol_usuario is None:
    st.title("⚕️ Portal Clínico UCI")
    if codigo_acceso:
        st.error("Código de acceso incorrecto.")
    else:
        st.info("Introduce tu código de acceso en el panel lateral para comenzar.")
        st.caption("Usa '007' para médico o '000' para familiar.")
else:
    st.title(f"⚕️ Asistente UCI - Modo: {rol_usuario}")

    with st.sidebar:
        st.subheader("🔍 Localización de Paciente")
        input_id = st.text_input("ID Paciente", "P0003122")

    datos_paciente = consultar_datos_paciente(input_id)

    if datos_paciente:
        if rol_usuario == "Personal Médico":
            st.success(f"Ficha médica activa: Paciente {datos_paciente['patient_id']}")

            col1, col2, col3 = st.columns(3)
            with col1:
                riesgo = datos_paciente.get('probabilidad_reingreso', 0)
                riesgo_pct = riesgo * 100
                if riesgo_pct >= 50:
                    icono = "🔴"
                elif riesgo_pct >= 30:
                    icono = "🟠"
                elif riesgo_pct >= 15:
                    icono = "🟡"
                else:
                    icono = "🟢"
                etiqueta = f"{icono} {riesgo_pct:.1f}%" if riesgo > 0 else "N/A"
                st.metric("Riesgo de Reingreso 30d", etiqueta)
            with col2:
                st.metric("Edad", f"{datos_paciente.get('edad', '?')} años")
            with col3:
                st.metric("Índice Charlson", datos_paciente.get('charlson', '?'))

            with st.expander("📋 Diagnósticos", expanded=True):
                st.write(datos_paciente.get('diagnosticos', 'Sin datos'))

            contexto_paciente = (
                f"Eres un asistente clínico de la UCI. Paciente {datos_paciente['patient_id']}, "
                f"{datos_paciente.get('edad', '?')} años. Diagnósticos: {datos_paciente.get('diagnosticos', 'N/A')}. "
                f"Índice Charlson: {datos_paciente.get('charlson', '?')}. "
                f"Riesgo de reingreso: {riesgo*100:.1f}%." if riesgo > 0 else "Riesgo de reingreso: N/A."
            )

        elif rol_usuario == "Familiar":
            st.info(f"Paciente {datos_paciente['patient_id']}")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Edad", f"{datos_paciente.get('edad', '?')} años")
            with col2:
                est = datos_paciente.get('dias_estancia', '?')
                st.metric("Estancia hospitalaria", f"{est} días" if est != "?" else "N/A")

            contexto_paciente = (
                f"Eres un asistente de atención hospitalaria. Hablas con un familiar del paciente "
                f"{datos_paciente['patient_id']}, {datos_paciente.get('edad', '?')} años. "
                "Sé amable y empático. No des diagnósticos específicos ni datos técnicos."
            )

        # ── CHAT ÚNICO (clínico + datos) ──
        st.markdown("---")
        st.subheader("💬 Asistente Virtual")

        # System prompt unificado: contexto paciente + esquema BD para consultas
        system_unificado = contexto_paciente + """

Además, si la pregunta requiere datos concretos de la base de datos (analíticas, medicamentos, diagnósticos, ingresos, etc.),
puedes generar una consulta SQL. Las tablas disponibles en DuckDB son:

- patients: patient_id, age, sex, bmi, charlson_index, dx_* (comorbilidades)
- outcomes: patient_id, admission_date, discharge_date, length_of_stay_days, icu_admission, readmitted_30d, total_charges_usd
- lab_results: patient_id, test_date, test_name, value, unit, flag, is_abnormal
- medications: patient_id, medication, dose, unit, frequency, indication, start_date, adherence_pct
- diagnoses: patient_id, visit_date, visit_type, primary_diagnosis, primary_icd10

Rutas: read_parquet('abfss://medicalproyect-processed@stproyectomastergrupo3.dfs.core.windows.net/silver/TABLA/*.parquet')

Si generas SQL, escríbelo entre ```sql y ```. Yo lo ejecutaré y devolveré los resultados.
Responde primero con texto, luego el SQL si procede.
SIEMPRE LIMITA LOS RESULTADOS A 50 FILAS."""

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            if msg["role"] != "system":
                with st.chat_message(msg["role"]):
                    content = msg["content"]
                    if isinstance(content, dict):
                        st.markdown(content.get("text", ""))
                        if "df" in content:
                            st.dataframe(content["df"], width='stretch')
                    else:
                        st.markdown(content)

        if prompt := st.chat_input("Escribe tu consulta sobre el paciente..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            mensajes = [{"role": "system", "content": system_unificado}] + [
                m for m in st.session_state.messages
                if isinstance(m["content"], str) and m["role"] != "system"
            ]

            with st.chat_message("assistant"):
                try:
                    respuesta_raw = client.chat.completions.create(
                        model=MODELO_LLM, messages=mensajes, temperature=0.2, max_tokens=1500
                    )
                    respuesta = respuesta_raw.choices[0].message.content

                    # Separar SQL del texto
                    import re as _re
                    sql_match = _re.search(r"```sql\n?(.*?)```", respuesta, _re.DOTALL)

                    if sql_match and modo_datos == "Text-to-SQL + ADLS (DuckDB)":
                        texto_previo = respuesta[:sql_match.start()].strip()
                        sql = sql_match.group(1).strip()

                        if texto_previo:
                            st.markdown(texto_previo)

                        st.code(sql, language="sql")

                        with st.spinner("Ejecutando consulta..."):
                            df = ejecutar_sql(sql)

                        if df is not None and len(df) > 0:
                            st.dataframe(df, width='stretch')
                            csv = df.to_csv(index=False).encode("utf-8")
                            st.download_button("📥 Descargar CSV", csv, "resultados.csv", "text/csv")
                            st.caption(f"{len(df)} filas")
                        elif df is not None:
                            st.info("Consulta ejecutada sin resultados.")
                        else:
                            st.warning("No se pudo ejecutar el SQL generado.")

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": {"text": respuesta, "df": df if df is not None and len(df) > 0 else None}
                        })
                    else:
                        st.markdown(respuesta)
                        st.session_state.messages.append({"role": "assistant", "content": respuesta})

                except Exception as e:
                    st.error(f"Error con {backend_seleccionado}: {e}")
    else:
        st.warning("Paciente no localizado.")
