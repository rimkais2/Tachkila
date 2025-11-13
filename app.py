
import streamlit as st
import pandas as pd
import uuid
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from sqlalchemy import (
    create_engine, MetaData, Table, Column, String, Integer, ForeignKey,
    select, insert, update, UniqueConstraint, delete, func
)
from sqlalchemy.engine import Engine

import random


# -----------------------------
# SESSION STATE INIT
# -----------------------------
if "player" not in st.session_state:
    st.session_state["player"] = None
if "admin_authenticated" not in st.session_state:
    st.session_state["admin_authenticated"] = False
if "collapse_sidebar" not in st.session_state:
    st.session_state["collapse_sidebar"] = False

# -----------------------------
# CONFIG PAGE
# -----------------------------
sidebar_state = "expanded" if not st.session_state["collapse_sidebar"] else "collapsed"

st.set_page_config(
    page_title="Tachkila Mouchkila",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state=sidebar_state,
)
# Secrets attendus
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "changeme")
DATABASE_URL = st.secrets.get("DATABASE_URL", "sqlite:///pronos.db")

# -----------------------------
# DB INIT
# -----------------------------
engine: Engine = create_engine(DATABASE_URL, future=True)
meta = MetaData()

users = Table(
    "users", meta,
    Column("user_id", String, primary_key=True),
    Column("display_name", String, unique=True, nullable=False),
    Column("pin_code", String, nullable=False),
    Column("is_game_master", Integer, nullable=False, server_default="0"),  # 0 = joueur, 1 = maître de jeu
)

matches = Table(
    "matches", meta,
    Column("match_id", String, primary_key=True),
    Column("home", String, nullable=False),
    Column("away", String, nullable=False),
    Column("kickoff_paris", String, nullable=False),  # "YYYY-MM-DD HH:MM" heure de Paris
    Column("final_home", Integer, nullable=True),
    Column("final_away", Integer, nullable=True),
    Column("category", String, nullable=True),  # 👉 nouvelle colonne
)


predictions = Table(
    "predictions", meta,
    Column("prediction_id", String, primary_key=True),
    Column("user_id", String, ForeignKey("users.user_id"), nullable=False),
    Column("match_id", String, ForeignKey("matches.match_id"), nullable=False),
    Column("ph", Integer, nullable=False),
    Column("pa", Integer, nullable=False),
    Column("timestamp_utc", String, nullable=False),
    UniqueConstraint("user_id", "match_id", name="uniq_user_match"),
)

with engine.begin() as conn:
    try:
        info = conn.exec_driver_sql("PRAGMA table_info(matches)").fetchall()
        existing_cols = [c[1] for c in info]
        if "category" not in existing_cols:
            conn.exec_driver_sql(
                "ALTER TABLE matches ADD COLUMN category TEXT"
            )
    except Exception:
        # Si pas SQLite, on ignore silencieusement
        pass


def init_first_user():
    """Crée un premier user par défaut si la table est vide."""
    with engine.begin() as conn:
        count = conn.execute(
            select(func.count()).select_from(users)
        ).scalar()
        if count == 0:
            uid = str(uuid.uuid4())
            display_name = "Joueur1"
            pin_code = "0000"
            conn.execute(
                insert(users).values(
                    user_id=uid,
                    display_name=display_name,
                    pin_code=pin_code,
                    is_game_master=0,
                )
            )

init_first_user()

# -----------------------------
# UTILS
# -----------------------------
def now_paris():
    return datetime.now(ZoneInfo("Europe/Paris"))

def is_editable(kickoff_paris_str: str) -> bool:
    """True si on peut encore modifier le prono (avant coup d'envoi)."""
    try:
        ko_local = datetime.strptime(
            kickoff_paris_str, "%Y-%m-%d %H:%M"
        ).replace(tzinfo=ZoneInfo("Europe/Paris"))
        return now_paris() < ko_local
    except Exception:
        return False

def result_sign(h, a):
    h, a = int(h), int(a)
    return (h > a) - (h < a)  # 1/0/-1

def compute_points(ph, pa, fh, fa):
    try:
        if fh is None or fa is None:
            return 0
        ph, pa, fh, fa = int(ph), int(pa), int(fh), int(fa)
        if ph == fh and pa == fa:
            return 4
        return 2 if result_sign(ph, pa) == result_sign(fh, fa) else 0
    except Exception:
        return 0

@st.cache_data(ttl=10)
def load_df():
    with engine.begin() as conn:
        df_users = pd.read_sql(select(users), conn)
        df_matches = pd.read_sql(select(matches), conn)
        df_preds = pd.read_sql(select(predictions), conn)
    return df_users, df_matches, df_preds

def upsert_prediction(user_id: str, match_id: str, ph: int, pa: int):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with engine.begin() as conn:
        row = conn.execute(
            select(predictions)
            .where(predictions.c.user_id == user_id, predictions.c.match_id == match_id)
        ).mappings().first()
        if row:
            conn.execute(
                update(predictions)
                .where(predictions.c.prediction_id == row["prediction_id"])
                .values(ph=int(ph), pa=int(pa), timestamp_utc=ts)
            )
        else:
            conn.execute(
                insert(predictions).values(
                    prediction_id=str(uuid.uuid4()),
                    user_id=user_id,
                    match_id=match_id,
                    ph=int(ph),
                    pa=int(pa),
                    timestamp_utc=ts,
                )
            )
    st.cache_data.clear()

def add_match(home: str, away: str, kickoff_paris: str, category: str | None = None):
    """Ajoute un match. kickoff_paris = 'YYYY-MM-DD HH:MM' heure de Paris."""
    _ = datetime.strptime(kickoff_paris, "%Y-%m-%d %H:%M")  # validation simple

    if category is not None:
        category = category.strip()
        if category == "":
            category = None

    with engine.begin() as conn:
        conn.execute(insert(matches).values(
            match_id=str(uuid.uuid4()),
            home=home.strip(),
            away=away.strip(),
            kickoff_paris=kickoff_paris.strip(),
            final_home=None,
            final_away=None,
            category=category,
        ))
    st.cache_data.clear()


def set_final_score(match_id: str, fh: int, fa: int):
    with engine.begin() as conn:
        conn.execute(
            update(matches)
            .where(matches.c.match_id == match_id)
            .values(final_home=int(fh), final_away=int(fa))
        )
    st.cache_data.clear()

def create_player(display_name: str) -> str:
    """Crée un joueur avec un code à 4 chiffres et renvoie ce code."""
    display_name = display_name.strip()
    if not display_name:
        raise ValueError("Le nom du joueur est obligatoire.")

    pin = f"{random.randint(1000, 9999)}"  # code aléatoire 4 chiffres

    with engine.begin() as conn:
        row = conn.execute(
            select(users).where(users.c.display_name == display_name)
        ).mappings().first()
        if row:
            raise ValueError("Ce joueur existe déjà.")

        uid = str(uuid.uuid4())
        conn.execute(
            insert(users).values(
                user_id=uid,
                display_name=display_name,
                pin_code=pin,
                is_game_master=0,
            )
        )

    st.cache_data.clear()
    return pin

def authenticate_player(display_name: str, pin_code: str):
    """Vérifie nom + code, renvoie le user ou None."""
    display_name = display_name.strip()
    pin_code = pin_code.strip()
    if not display_name or not pin_code:
        return None

    with engine.begin() as conn:
        row = conn.execute(
            select(users).where(
                users.c.display_name == display_name,
                users.c.pin_code == pin_code
            )
        ).mappings().first()
    return row  # dict-like ou None

def delete_match_and_predictions(match_id: str):
    """Supprime un match et tous les pronostics associés."""
    with engine.begin() as conn:
        conn.execute(delete(predictions).where(predictions.c.match_id == match_id))
        conn.execute(delete(matches).where(matches.c.match_id == match_id))
    st.cache_data.clear()

def set_game_master(user_id: str, is_gm: bool):
    """Active ou désactive le rôle maître de jeu pour un joueur."""
    with engine.begin() as conn:
        conn.execute(
            update(users)
            .where(users.c.user_id == user_id)
            .values(is_game_master=1 if is_gm else 0)
        )
    st.cache_data.clear()

@st.cache_data
def load_catalog():
    """Charge la liste des clubs et sélections depuis le CSV."""
    return pd.read_csv("teams_catalog.csv")

catalog = load_catalog()

def logo_for(team_name):
    """Retourne le lien du logo si disponible."""
    try:
        row = catalog.loc[catalog["name"] == team_name]
        if row.empty:
            return None
        url = row.iloc[0]["logo_url"]
        if isinstance(url, str) and len(url) > 0:
            return url
    except Exception:
        return None
    return None

# -----------------------------
# UI - SIDEBAR
# -----------------------------
st.title("⚽ Tachkila Mouchkila")

with st.sidebar:
    # Connexion joueur
    st.header("Connexion joueur")

    if st.session_state["player"] is None:
        name_input = st.text_input("Nom du joueur")
        pin_input = st.text_input("Code à 4 chiffres", type="password", max_chars=4)

        if st.button("Se connecter"):
            user = authenticate_player(name_input, pin_input)
            if user is None:
                st.error("Nom ou code incorrect (demande à l'admin de vérifier ton code).")
            else:
                st.session_state["player"] = dict(user)
                st.session_state["collapse_sidebar"] = True   # 👈 replie la sidebar
                st.rerun()

    else:
        player = st.session_state["player"]
        st.success(f"Connecté : {player['display_name']}")
        if st.button("Changer de joueur"):
            st.session_state["player"] = None
            st.rerun()

    st.markdown("---")

    # Mode admin
    st.header("Mode administrateur")

    if not st.session_state["admin_authenticated"]:
        admin_pw_input = st.text_input("Mot de passe admin", type="password")
        if st.button("Activer le mode admin"):
            if admin_pw_input == ADMIN_PASSWORD:
                st.session_state["admin_authenticated"] = True
                st.success("Mode admin activé")
                st.rerun()
            else:
                st.error("Mot de passe incorrect.")
    else:
        st.success("Mode admin actif")
        if st.button("Désactiver le mode admin"):
            st.session_state["admin_authenticated"] = False
            st.rerun()

# -----------------------------
# CONTEXTE UTILISATEUR
# -----------------------------
player = st.session_state["player"]
admin_authenticated = st.session_state["admin_authenticated"]

if player is None:
    st.info("Commence par te connecter avec ton nom + code à 4 chiffres dans la colonne de gauche.")
    st.stop()

df_users, df_matches, df_preds = load_df()
user_id = player["user_id"]
display_name = player["display_name"]

# Rôle maître de jeu ?
row_me = df_users[df_users["user_id"] == user_id]
if not row_me.empty and "is_game_master" in row_me.columns:
    is_game_master = bool(row_me.iloc[0]["is_game_master"])
else:
    is_game_master = False

can_manage_matches = admin_authenticated or is_game_master

# -----------------------------
# TABS
# -----------------------------
# -----------------------------
# TABS (créés dynamiquement selon le rôle)
# -----------------------------
tab_labels = ["Pronostiquer", "Classement"]
tab_ids = ["pronos", "classement"]

# Onglet "Maître de jeu" visible pour admin OU maître de jeu
if can_manage_matches:  # can_manage_matches = admin ou game master
    tab_labels.append("Maître de jeu")
    tab_ids.append("maitre")

# Onglet "Admin" visible uniquement pour l'admin
if admin_authenticated:
    tab_labels.append("Admin")
    tab_ids.append("admin")

tabs = st.tabs(tab_labels)
tab_dict = dict(zip(tab_ids, tabs))

tab_pronos = tab_dict["pronos"]
tab_classement = tab_dict["classement"]
tab_maitre = tab_dict.get("maitre")   # peut être None si pas autorisé
tab_admin = tab_dict.get("admin")     # peut être None si pas admin


# -----------------------------
# TAB PRONOS
# -----------------------------
with tab_pronos:
    st.subheader("Mes pronostics")

    if df_matches.empty:
        st.info("Aucun match pour le moment.")
    else:
        try:
            df_matches_sorted = df_matches.copy()
            df_matches_sorted["_ko"] = pd.to_datetime(df_matches_sorted["kickoff_paris"], format="%Y-%m-%d %H:%M")
        except Exception:
            df_matches_sorted = df_matches.copy()
            df_matches_sorted["_ko"] = pd.NaT

        df_matches_sorted = df_matches_sorted.sort_values("_ko", ascending=False, na_position="last").drop(columns=["_ko"])


        my_preds = df_preds[df_preds["user_id"] == user_id]

        for _, m in df_matches_sorted.iterrows():
            st.markdown("---")
            c1, c2, c3, c4 = st.columns([3,3,3,2])

            # Infos match + logos
            with c1:
                l1, l2, l3 = st.columns([1,2,1])
                with l1:
                    lg_home = logo_for(m["home"])
                    if lg_home:
                        st.image(lg_home, width=40)
                with l2:
                    st.markdown(f"**{m['home']} vs {m['away']}**")
                    st.caption(f"Coup d’envoi : {m['kickoff_paris']} (heure de Paris)")
                with l3:
                    lg_away = logo_for(m["away"])
                    if lg_away:
                        st.image(lg_away, width=40)

            existing = my_preds[my_preds["match_id"] == m["match_id"]]
            ph0 = int(existing.iloc[0]["ph"]) if not existing.empty else 0
            pa0 = int(existing.iloc[0]["pa"]) if not existing.empty else 0

            editable = is_editable(m["kickoff_paris"])
            res_known = (pd.notna(m["final_home"]) and pd.notna(m["final_away"]))

            with c2:
                ph = st.number_input(
                    f"{m['home']} (dom.)",
                    0, 20, ph0, 1,
                    key=f"ph_{m['match_id']}",
                    disabled=not editable
                )
            with c3:
                pa = st.number_input(
                    f"{m['away']} (ext.)",
                    0, 20, pa0, 1,
                    key=f"pa_{m['match_id']}",
                    disabled=not editable
                )
            with c4:
                if editable:
                    if st.button("💾 Enregistrer", key=f"save_{m['match_id']}"):
                        upsert_prediction(user_id, m["match_id"], ph, pa)
                        st.success("Pronostic enregistré ✅")

                else:
                    st.info("⛔ Verrouillé (match commencé)")

            if res_known and not editable:
                st.caption(f"Score final : {int(m['final_home'])} - {int(m['final_away'])}")

# -----------------------------
# TAB CLASSEMENT
# -----------------------------
with tab_classement:
    st.subheader("Classement général")

    if df_preds.empty or df_matches.empty:
        st.info("Pas encore de pronostics ou de matches terminés.")
    else:
        merged = (
            df_preds
            .merge(df_matches, on="match_id", how="left")
            .merge(df_users, on="user_id", how="left")
        )

        merged["points"] = merged.apply(
            lambda r: compute_points(r["ph"], r["pa"], r["final_home"], r["final_away"]),
            axis=1
        )

        leaderboard = (
            merged.groupby(["user_id", "display_name"], dropna=False)["points"]
            .sum()
            .reset_index()
        )
        leaderboard = leaderboard.sort_values(
            ["points", "display_name"], ascending=[False, True]
        )

        if leaderboard.empty:
            st.info("Les scores finaux ne sont pas encore saisis.")
        else:
            st.markdown("### Podium")

            top3 = leaderboard.head(3).reset_index(drop=True)
            cols = st.columns(3)

            medals = ["🥇", "🥈", "🥉"]
            colors = ["#ffd700", "#c0c0c0", "#cd7f32"]

            for i, row in top3.iterrows():
                with cols[i]:
                    pseudo = row["display_name"]
                    pts = row["points"]
                    medal = medals[i]
                    color = colors[i]

                    st.markdown(
                        f"""
                        <div style="
                            background:{color}22;
                            border:2px solid {color};
                            border-radius:20px;
                            padding:16px;
                            text-align:center;
                            box-shadow:0 4px 10px rgba(0,0,0,0.15);
                        ">
                            <div style="font-size:40px;">{medal}</div>
                            <div style="font-size:20px;font-weight:700;margin-top:4px;">{pseudo}</div>
                            <div style="font-size:16px;margin-top:8px;">{pts} points</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown("### Classement complet")

            lb = leaderboard.reset_index(drop=True)
            for idx, row in lb.iterrows():
                rank = idx + 1
                pseudo = row["display_name"]
                pts = row["points"]

                st.markdown(
                    f"""
                    <div style="
                        display:flex;
                        align-items:center;
                        margin-bottom:6px;
                    ">
                        <div style="
                            width:32px;height:32px;
                            border-radius:50%;
                            background:#0f4c81;
                            color:white;
                            display:flex;
                            align-items:center;
                            justify-content:center;
                            font-weight:700;
                            margin-right:8px;
                        ">{rank}</div>
                        <div style="flex:1;">
                            <span style="font-weight:600;">{pseudo}</span>
                            <span style="color:#555;"> — {pts} pts</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with st.expander("Détail par match"):
                show = merged[
                    [
                        "display_name",
                        "home", "away",
                        "ph", "pa",
                        "final_home", "final_away",
                        "points",
                        "kickoff_paris",
                    ]
                ].copy()

                show = show.rename(
                    columns={
                        "display_name": "Joueur",
                        "home": "Domicile",
                        "away": "Extérieur",
                        "ph": "Prono D",
                        "pa": "Prono E",
                        "final_home": "Final D",
                        "final_away": "Final E",
                        "points": "Pts",
                        "kickoff_paris": "Coup d’envoi",
                    }
                )

                st.dataframe(show, use_container_width=True)


# -----------------------------
# TAB MAÎTRE DE JEU
# -----------------------------
if tab_maitre is not None:
    with tab_maitre:
        st.subheader("Espace maître de jeu")

        if not can_manage_matches:
            st.info("Réservé à l'administrateur ou aux maîtres de jeu.")
        else:
            # Bandeau d'info sur le rôle
            if admin_authenticated and is_game_master:
                st.success("Mode admin + maître de jeu actifs.")
            elif admin_authenticated:
                st.success("Mode admin actif.")
            elif is_game_master:
                st.success("Mode maître de jeu actif (gestion des matches et des pronos des joueurs).")

            # -------------------------
            # SOUS-ONGLETS
            # -------------------------
            tab_ajout, tab_resultats, tab_pronos_joueurs = st.tabs(
                ["Ajouter un match", "Résultats", "Pronos joueurs"]
            )

            # =====================================================
            # ONGLET 1 : AJOUTER UN MATCH
            # =====================================================
            with tab_ajout:
    st.markdown("### ➕ Ajouter un match")

    # Charger les catégories existantes
    df_users_cat, df_matches_cat, _ = load_df()
    existing_categories: list[str] = []
    if "category" in df_matches_cat.columns:
        existing_categories = sorted(
            [
                str(c).strip()
                for c in df_matches_cat["category"].dropna().unique()
                if str(c).strip() != ""
            ]
        )

    # Options du selectbox
    options = ["(Aucune catégorie)"]
    if existing_categories:
        options += existing_categories
    options.append("➕ Nouvelle catégorie...")

    cat_choice = st.selectbox("Catégorie du match (optionnel)", options)
    new_cat = ""
    if cat_choice == "➕ Nouvelle catégorie...":
        new_cat = st.text_input("Nouvelle catégorie", placeholder="Ex : Poules, Quart de finale, Match amical...")

    with st.form("form_add_match"):
        c1, c2, c3, c4 = st.columns([3, 3, 3, 2])

        with c1:
            home = st.selectbox(
                "Équipe domicile",
                options=catalog["name"].sort_values(),
                index=None,
                placeholder="Choisir une équipe..."
            )
            if home:
                logo = logo_for(home)
                if logo:
                    st.image(logo, width=64, caption=home)

        with c2:
            away = st.selectbox(
                "Équipe extérieur",
                options=catalog["name"].sort_values(),
                index=None,
                placeholder="Choisir une équipe..."
            )
            if away:
                logo = logo_for(away)
                if logo:
                    st.image(logo, width=64, caption=away)

        with c3:
            col_date, col_time = st.columns(2)
            with col_date:
                date_match = st.date_input("📅 Date du match")
            with col_time:
                heure_match = st.time_input("⏰ Heure du match")
            kickoff_dt = datetime.combine(date_match, heure_match)
            kickoff = kickoff_dt.strftime("%Y-%m-%d %H:%M")

        with c4:
            submit = st.form_submit_button("Ajouter")

        if submit:
            if not home or not away:
                st.warning("Sélectionne les deux équipes.")
            elif home == away:
                st.warning("L'équipe domicile et l'équipe extérieur doivent être différentes.")
            else:
                # Déterminer la catégorie finale
                if new_cat.strip():
                    category = new_cat.strip()
                elif cat_choice not in ["(Aucune catégorie)", "➕ Nouvelle catégorie..."]:
                    category = cat_choice
                else:
                    category = None

                add_match(home, away, kickoff, category)
                if category:
                    st.success(f"Match ajouté ✅ ({home} vs {away} — {kickoff}, catégorie : {category})")
                else:
                    st.success(f"Match ajouté ✅ ({home} vs {away} — {kickoff})")
                st.rerun()


            # =====================================================
            # ONGLET 2 : RÉSULTATS
            # =====================================================
            with tab_resultats:
                st.markdown("### 📝 Saisie et modification des résultats")

                df_users3, df_matches3, _ = load_df()
                if df_matches3.empty:
                    st.info("Aucun match pour le moment.")
                else:
                    # Tri du plus récent au plus ancien
                    try:
                        df_matches3["_ko"] = pd.to_datetime(
                            df_matches3["kickoff_paris"], format="%Y-%m-%d %H:%M"
                        )
                    except Exception:
                        df_matches3["_ko"] = pd.NaT

                    df_matches3 = df_matches3.sort_values(
                        "_ko", ascending=False, na_position="last"
                    ).drop(columns=["_ko"])

                    for _, m in df_matches3.iterrows():
                        match_id = m["match_id"]

                        with st.expander(f"{m['home']} vs {m['away']} — {m['kickoff_paris']}"):
                            c1, c2 = st.columns([3, 2])

                            # Infos générales + logos
                            with c1:
                                st.markdown(f"**{m['home']} vs {m['away']}**")
                                st.caption(f"Coup d’envoi : {m['kickoff_paris']} (heure de Paris)")

                                lc1, lc2 = st.columns(2)
                                with lc1:
                                    lg_home = logo_for(m["home"])
                                    if lg_home:
                                        st.image(lg_home, width=48, caption=m["home"])
                                with lc2:
                                    lg_away = logo_for(m["away"])
                                    if lg_away:
                                        st.image(lg_away, width=48, caption=m["away"])

                            # Score actuel
                            with c2:
                                if pd.notna(m["final_home"]) and pd.notna(m["final_away"]):
                                    st.markdown(
                                        f"**Score final actuel :** {int(m['final_home'])} - {int(m['final_away'])}"
                                    )
                                else:
                                    st.markdown("**Score final actuel :** non saisi")

                            st.markdown("---")

                            # Zone de saisie du score + actions
                            c3, c4, c5 = st.columns([2, 2, 2])

                            default_fh = int(m["final_home"]) if pd.notna(m["final_home"]) else 0
                            default_fa = int(m["final_away"]) if pd.notna(m["final_away"]) else 0

                            with c3:
                                new_fh = st.number_input(
                                    f"Score {m['home']}",
                                    min_value=0,
                                    max_value=50,
                                    step=1,
                                    value=default_fh,
                                    key=f"fh_admin_{match_id}"
                                )
                            with c4:
                                new_fa = st.number_input(
                                    f"Score {m['away']}",
                                    min_value=0,
                                    max_value=50,
                                    step=1,
                                    value=default_fa,
                                    key=f"fa_admin_{match_id}"
                                )

                            with c5:
                                if st.button("💾 Sauvegarder le score", key=f"save_score_{match_id}"):
                                    set_final_score(match_id, new_fh, new_fa)
                                    st.success("Score final mis à jour ✅ (le classement sera recalculé)")
                                    st.rerun()

                                if st.button("🗑️ Supprimer ce match", key=f"delete_match_{match_id}"):
                                    delete_match_and_predictions(match_id)
                                    st.warning("Match supprimé avec ses pronostics associés 🗑️")
                                    st.rerun()

            # =====================================================
            # ONGLET 3 : PRONOS DES JOUEURS
            # =====================================================
            with tab_pronos_joueurs:
                st.markdown("### ✍️ Saisir ou corriger les pronostics d'un joueur")

                # Sélection du joueur dont on modifie les pronos
                joueurs = df_users.sort_values("display_name").reset_index(drop=True)
                if joueurs.empty:
                    st.info("Aucun joueur.")
                else:
                    choix_joueur = st.selectbox(
                        "Choisir un joueur :",
                        joueurs["display_name"].tolist(),
                    )
                    cible = joueurs[joueurs["display_name"] == choix_joueur].iloc[0]
                    target_user_id = cible["user_id"]

                    st.caption(f"Modification des pronostics pour : **{choix_joueur}**")

                    if df_matches.empty:
                        st.info("Aucun match pour le moment.")
                    else:
                        # Tri du plus récent au plus ancien
                        try:
                            df_matches_gm = df_matches.copy()
                            df_matches_gm["_ko"] = pd.to_datetime(
                                df_matches_gm["kickoff_paris"], format="%Y-%m-%d %H:%M"
                            )
                        except Exception:
                            df_matches_gm = df_matches.copy()
                            df_matches_gm["_ko"] = pd.NaT

                        df_matches_gm = df_matches_gm.sort_values(
                            "_ko", ascending=False, na_position="last"
                        ).drop(columns=["_ko"])

                        preds_cible = df_preds[df_preds["user_id"] == target_user_id]

                        for _, m in df_matches_gm.iterrows():
                            st.markdown("---")
                            c1, c2, c3, c4 = st.columns([3, 3, 3, 2])

                            # Infos match
                            with c1:
                                st.markdown(f"**{m['home']} vs {m['away']}**")
                                st.caption(f"Coup d’envoi : {m['kickoff_paris']} (heure de Paris)")

                            # Prono existant
                            existing = preds_cible[preds_cible["match_id"] == m["match_id"]]
                            ph0 = int(existing.iloc[0]["ph"]) if not existing.empty else 0
                            pa0 = int(existing.iloc[0]["pa"]) if not existing.empty else 0
                            
                            # Pour le maître de jeu : toujours éditable, même si le match a commencé
                            res_known = (pd.notna(m["final_home"]) and pd.notna(m["final_away"]))
                            
                            with c2:
                                ph = st.number_input(
                                    f"{m['home']} (dom.)",
                                    0, 20, ph0, 1,
                                    key=f"gm_ph_{target_user_id}_{m['match_id']}",
                                    disabled=False,  # 👈 jamais désactivé pour le maître de jeu
                                )
                            with c3:
                                pa = st.number_input(
                                    f"{m['away']} (ext.)",
                                    0, 20, pa0, 1,
                                    key=f"gm_pa_{target_user_id}_{m['match_id']}",
                                    disabled=False,  # 👈 idem
                                )
                            
                            with c4:
                                if st.button("💾 Enregistrer", key=f"gm_save_{target_user_id}_{m['match_id']}"):
                                    upsert_prediction(target_user_id, m["match_id"], ph, pa)
                                    st.success(f"Pronostic enregistré pour {choix_joueur} ✅")
                            
                            if res_known:
                                st.caption(f"Score final : {int(m['final_home'])} - {int(m['final_away'])}")


# -----------------------------
# TAB ADMIN (gestion joueurs & rôles)
# -----------------------------
if tab_admin is not None:
    with tab_admin:
        st.subheader("Administration des joueurs")

    if not admin_authenticated:
        st.info("Réservé à l'administrateur. Active le mode admin dans la barre latérale.")
    else:
        st.success("Mode admin actif")

        # Ajout joueur
        st.markdown("### Ajouter un nouveau joueur")

        with st.form("add_player"):
            new_player_name = st.text_input("Nom du joueur (ex: Karim)")
            submit_player = st.form_submit_button("Créer le joueur")

        if submit_player:
            try:
                pin = create_player(new_player_name)
                st.success(f"Joueur créé — Nom : {new_player_name} — Code : {pin}")
                st.info("Note ce code et communique-le au joueur.")
            except ValueError as e:
                st.error(str(e))

        st.markdown("---")

        # Liste joueurs + rôle
        st.markdown("### Joueurs existants et rôles")

        df_users4, _, _ = load_df()
        if df_users4.empty:
            st.write("Aucun joueur créé pour l'instant.")
        else:
            if "is_game_master" not in df_users4.columns:
                df_users4["is_game_master"] = 0

            for _, row in df_users4.sort_values("display_name").iterrows():
                user_id_row = row["user_id"]
                name = row["display_name"]
                pin = row["pin_code"]
                is_gm = bool(row["is_game_master"])

                c1, c2, c3, c4 = st.columns([3, 2, 2, 3])

                with c1:
                    st.markdown(f"**{name}**")
                with c2:
                    st.caption(f"Code : `{pin}`")
                with c3:
                    st.write("Maître de jeu :", "✅" if is_gm else "❌")
                with c4:
                    if is_gm:
                        if st.button("Retirer maître de jeu", key=f"unset_gm_{user_id_row}"):
                            set_game_master(user_id_row, False)
                            st.success(f"{name} n'est plus maître de jeu.")
                            st.rerun()
                    else:
                        if st.button("Nommer maître de jeu", key=f"set_gm_{user_id_row}"):
                            set_game_master(user_id_row, True)
                            st.success(f"{name} est maintenant maître de jeu.")
                            st.rerun()
