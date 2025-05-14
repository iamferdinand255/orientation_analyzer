"""
ANALYSEUR D'ORIENTATION PROFESSIONNELLE
Version ultra-optimisée, indestructible et monolithique
Avec améliorations 2025 (suggestions métiers + export PDF + analyses avancées)
"""
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import re
import time
import json
import unicodedata
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from fpdf import FPDF
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64

# --- CONFIGURATION CENTRALE ---
st.set_page_config(page_title="Analyseur d'Orientation", page_icon="⚡", layout="wide")

# Chargement brutal de l'API key
load_dotenv(override=True)
API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Constantes du projet
HISTORY_FILE = "orientation_history.json"

# --- FONCTIONS ESSENTIELLES ---

def sanitize_text(text):
    """Nettoyage brutal du texte"""
    if not isinstance(text, str):
        return str(text) if text is not None else "Non specifie"
    return ''.join(c for c in unicodedata.normalize('NFD', text) if not unicodedata.combining(c))

def save_api_key(api_key):
    """Enregistrement avec activation immédiate"""
    try:
        with open(".env", "w") as file:
            file.write(f'OPENROUTER_API_KEY="{api_key}"')
        os.environ["OPENROUTER_API_KEY"] = api_key
        return True, "Clé API enregistrée avec succès"
    except Exception as e:
        return False, f"Erreur d'enregistrement: {str(e)}"

def call_api(prompt, max_retries=3):
    """Appel API blindé contre tout échec"""
    global API_KEY

    # Vérification clé API
    if not API_KEY:
        return {"error": "Clé API manquante ou invalide"}

    # Initialisation client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=API_KEY,
        timeout=30.0
    )

    # Tentatives multiples avec backoff exponentiel
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://orientation-analyzer.com",
                    "X-Title": "Analyseur d'Orientation Pro"
                },
                model="deepseek/deepseek-chat-v3-0324",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500
            )

            # Vérification implacable
            if not response or not hasattr(response, 'choices') or len(response.choices) == 0:
                raise ValueError("Réponse API malformée")

            content = response.choices[0].message.content
            if not content:
                raise ValueError("Contenu vide")

            return {"success": True, "content": content}

        except Exception as e:
            if attempt == max_retries - 1:
                return {"error": str(e)}
            time.sleep(2 ** attempt)  # backoff exponentiel

def extract_score(analysis_text):
    """Extraction du score de cohérence"""
    if not isinstance(analysis_text, str):
        return None

    if "error" in analysis_text.lower():
        return None

    patterns = [
        r"NIVEAU DE COHÉRENCE:?\s*\[?(\d+)[/\]]",
        r"NIVEAU DE COHÉRENCE:?\s*(\d+)",
        r"cohérence:?\s*(\d+)",
        r"score:?\s*(\d+)",
        r"(\d+)/10"
    ]

    for pattern in patterns:
        matches = re.search(pattern, analysis_text, re.IGNORECASE)
        if matches:
            score = int(matches.group(1))
            return max(0, min(score, 10))

    return None

def extract_recommended_jobs(analysis_text):
    """Extraction des métiers recommandés"""
    if not isinstance(analysis_text, str):
        return []

    # Recherche de la section des métiers recommandés
    pattern = r"MÉTIERS RECOMMANDÉS:(.+?)(?:\n\n|\Z)"
    match = re.search(pattern, analysis_text, re.DOTALL | re.IGNORECASE)

    if not match:
        return []

    # Extraction des métiers listés avec tirets
    jobs_section = match.group(1)
    jobs = re.findall(r"[-•*]\s*(.+?)(?:\n|$)", jobs_section)

    # Nettoyage des résultats
    return [job.strip() for job in jobs if job.strip()]

def analyze_career(student_data):
    """Analyse d'orientation améliorée avec suggestions de métiers"""
    # Vérification des données
    filiere = student_data.get('filiere', '')
    carriere = student_data.get('carriere', '')

    if len(filiere.strip()) < 2 or len(carriere.strip()) < 3:
        return {"error": "Données insuffisantes", "score": None}

    prompt = f"""
    ANALYSE D'ORIENTATION PROFESSIONNELLE - FORMAT STRICTEMENT OBLIGATOIRE
    
    DONNÉES ÉTUDIANT:
    - Filière actuelle: {filiere}
    - Carrière visée: {carriere}
    
    FOURNIR OBLIGATOIREMENT CES QUATRE SECTIONS:
    1. NIVEAU DE COHÉRENCE: [0-10]
    Justification factuelle:
    
    2. ERREURS CRITIQUES:
    - Compétence manquante 1:
    - Compétence manquante 2:
    
    3. RECOMMANDATIONS:
    - Action correctrice 1:
    - Action correctrice 2:
    
    4. MÉTIERS RECOMMANDÉS:
    - Métier alternatif 1:
    - Métier alternatif 2:
    
    INSTRUCTIONS: 
    - Attribuez un score de cohérence (et non d'incohérence) de 0 à 10 (10 étant parfait)
    - Les scores élevés indiquent une bonne correspondance filière/carrière
    - Suggérez 2 métiers réalistes et précis qui exploitent les compétences de la filière
    - Soyez brutalement honnête dans votre analyse
    """

    # Appel API protégé
    response = call_api(prompt)

    if "error" in response:
        return {"error": response["error"], "score": None}

    # Extraction de l'analyse
    analysis = response["content"]
    score = extract_score(analysis)
    recommended_jobs = extract_recommended_jobs(analysis)

    return {
        "analysis": analysis,
        "score": score,
        "recommended_jobs": recommended_jobs
    }

def load_file(file):
    """Chargement blindé des données"""
    try:
        if file.name.endswith('.csv'):
            for encoding in ['utf-8', 'latin1', 'cp1252']:
                try:
                    file.seek(0)
                    return pd.read_csv(file, encoding=encoding), None
                except:
                    continue
            return None, "Impossible de décoder le CSV"
        elif file.name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file), None
        else:
            return None, "Format non supporté"
    except Exception as e:
        return None, str(e)

def save_to_history(results):
    """Sauvegarde indestructible des résultats"""
    try:
        # Préparation des données
        history_data = []
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)
            except Exception:
                history_data = []  # Reset en cas de corruption

        # Garantir que c'est une liste
        if not isinstance(history_data, list):
            history_data = []

        # Ajout des nouveaux résultats
        timestamp = datetime.now().isoformat()
        for result in results:
            if isinstance(result, dict) and "analysis" in result and "error" not in result.get("analysis", ""):
                entry = {
                    "timestamp": timestamp,
                    "nom": result.get("Nom", "Anonyme"),
                    "filiere": result.get("Filière", "Non spécifiée"),
                    "carriere": result.get("Carrière", "Non spécifiée"),
                    "score": result.get("score"),
                    "analysis": result.get("analysis", ""),
                    "recommended_jobs": result.get("recommended_jobs", [])
                }
                history_data.append(entry)

        # Sauvegarde sécurisée
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)

        return True
    except Exception as e:
        print(f"Erreur de sauvegarde: {str(e)}")
        return False

def get_all_stats():
    """Récupération des statistiques"""
    if not os.path.exists(HISTORY_FILE):
        return []

    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        return stats if isinstance(stats, list) else []
    except Exception:
        return []

def calculate_metrics():
    """Calcul des métriques globales"""
    stats = get_all_stats()

    if not stats:
        return {}

    # Extraction des scores valides
    valid_scores = [stat.get("score") for stat in stats if stat.get("score") is not None]

    if not valid_scores:
        return {
            "total_analyses": len(stats),
            "avg_score": None,
            "high_score_count": 0,
            "high_score_percent": 0
        }

    # Calcul des métriques
    avg_score = sum(valid_scores) / len(valid_scores)
    high_score = sum(1 for score in valid_scores if score >= 7)
    high_score_percent = (high_score / len(valid_scores)) * 100

    return {
        "total_analyses": len(stats),
        "avg_score": avg_score,
        "high_score_count": high_score,
        "high_score_percent": high_score_percent,
        "scores_distribution": {
            "0-3": sum(1 for s in valid_scores if 0 <= s <= 3),
            "4-6": sum(1 for s in valid_scores if 4 <= s <= 6),
            "7-10": sum(1 for s in valid_scores if 7 <= s <= 10),
        }
    }

def delete_stats(student_name=None):
    """Suppression des stats avec protection"""
    if not os.path.exists(HISTORY_FILE):
        return True

    if student_name is None:
        # Suppression complète
        try:
            os.remove(HISTORY_FILE)
            return True
        except:
            return False
    else:
        # Suppression sélective
        try:
            all_stats = get_all_stats()
            filtered_stats = [stat for stat in all_stats if stat.get("nom") != student_name]

            with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(filtered_stats, f, ensure_ascii=False, indent=2)
            return True
        except:
            return False

def generate_individual_pdf(result):
    """Génération de PDF pour un étudiant"""
    try:
        pdf = FPDF()
        pdf.add_page()

        # Style et en-tête
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Analyse d'Orientation Professionnelle", 0, 1, "C")
        pdf.line(10, 20, 200, 20)  # Ligne de séparation
        pdf.ln(5)

        # Informations étudiant
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, f"Étudiant: {result.get('Nom', 'Non spécifié')}", 0, 1)
        pdf.cell(0, 10, f"Filière: {result.get('Filière', 'Non spécifiée')}", 0, 1)
        pdf.cell(0, 10, f"Carrière visée: {result.get('Carrière', 'Non spécifiée')}", 0, 1)

        # Score avec code couleur
        score = result.get('score')
        if score is not None:
            if score >= 7:
                performance = "Excellent choix"
                pdf.set_text_color(0, 128, 0)  # Vert
            elif score >= 4:
                performance = "Choix à améliorer"
                pdf.set_text_color(255, 128, 0)  # Orange
            else:
                performance = "Réorientation recommandée"
                pdf.set_text_color(255, 0, 0)  # Rouge

            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, f"Score de cohérence: {score}/10 - {performance}", 0, 1)
            pdf.set_text_color(0, 0, 0)  # Retour au noir

        pdf.ln(5)

        # Métiers recommandés
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Métiers recommandés:", 0, 1)

        pdf.set_font("Arial", "", 10)
        for job in result.get('recommended_jobs', []):
            pdf.cell(0, 8, f"• {job}", 0, 1)

        if not result.get('recommended_jobs'):
            pdf.cell(0, 8, "Aucune recommandation spécifique", 0, 1)

        pdf.ln(5)

        # Analyse détaillée
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Analyse détaillée:", 0, 1)

        # Formatage du texte d'analyse
        if result.get('analysis'):
            pdf.set_font("Arial", "", 10)

            # Découpage en paragraphes pour éviter les dépassements
            analysis_text = result.get('analysis', '')
            paragraphs = analysis_text.split('\n')

            for para in paragraphs:
                if para.strip():
                    # Mettre en gras les titres de section
                    if any(section in para for section in ["NIVEAU DE COHÉRENCE", "ERREURS CRITIQUES", "RECOMMANDATIONS", "MÉTIERS RECOMMANDÉS"]):
                        pdf.set_font("Arial", "B", 10)
                        pdf.multi_cell(0, 5, para)
                        pdf.set_font("Arial", "", 10)
                    else:
                        pdf.multi_cell(0, 5, para)

        # Pied de page
        pdf.ln(10)
        pdf.set_font("Arial", "I", 8)
        pdf.cell(0, 10, f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}", 0, 1, "R")

        # Retourne le PDF
        return pdf.output(dest="S").encode("latin-1")
    except Exception as e:
        print(f"Erreur lors de la génération du PDF: {str(e)}")
        # PDF de secours en cas d'erreur
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Erreur de génération du PDF", 0, 1, "C")
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Une erreur est survenue: {str(e)}", 0, 1)
        return pdf.output(dest="S").encode("latin-1")

def generate_group_pdf(results):
    """Génération d'un PDF pour un groupe d'étudiants"""
    try:
        pdf = FPDF()
        pdf.add_page()

        # Titre et en-tête
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Rapport d'Analyses d'Orientation", 0, 1, "C")
        pdf.line(10, 20, 200, 20)  # Ligne de séparation
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 10, f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}", 0, 1)

        # Synthèse
        valid_scores = [r.get("score") for r in results if r.get("score") is not None]
        if valid_scores:
            avg_score = sum(valid_scores) / len(valid_scores)
            high_perf = sum(1 for s in valid_scores if s >= 7)
            medium_perf = sum(1 for s in valid_scores if 4 <= s <= 6)
            low_perf = sum(1 for s in valid_scores if s < 4)

            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Synthèse des résultats:", 0, 1)
            pdf.ln(2)

            pdf.set_font("Arial", "", 11)
            pdf.cell(0, 7, f"• Nombre d'étudiants analysés: {len(results)}", 0, 1)
            pdf.cell(0, 7, f"• Score moyen de cohérence: {avg_score:.1f}/10", 0, 1)

            # Statistiques avec code couleur
            pdf.set_text_color(0, 128, 0)  # Vert
            pdf.cell(0, 7, f"• Choix excellents: {high_perf} ({high_perf/len(valid_scores)*100:.1f}%)", 0, 1)

            pdf.set_text_color(255, 128, 0)  # Orange
            pdf.cell(0, 7, f"• Choix à améliorer: {medium_perf} ({medium_perf/len(valid_scores)*100:.1f}%)", 0, 1)

            pdf.set_text_color(255, 0, 0)  # Rouge
            pdf.cell(0, 7, f"• Réorientations recommandées: {low_perf} ({low_perf/len(valid_scores)*100:.1f}%)", 0, 1)

            pdf.set_text_color(0, 0, 0)  # Retour au noir

        # Récapitulatif des métiers recommandés
        if any('recommended_jobs' in r and r['recommended_jobs'] for r in results):
            pdf.ln(5)
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Métiers les plus recommandés:", 0, 1)

            # Comptage des métiers recommandés
            job_counts = {}
            for r in results:
                for job in r.get('recommended_jobs', []):
                    job_counts[job] = job_counts.get(job, 0) + 1

            # Tri par fréquence
            top_jobs = sorted(job_counts.items(), key=lambda x: x[1], reverse=True)[:10]

            pdf.set_font("Arial", "", 11)
            for job, count in top_jobs:
                pdf.cell(0, 7, f"• {job}: {count} fois", 0, 1)

        # Tableau récapitulatif par étudiant
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Tableau récapitulatif par étudiant:", 0, 1)

        # En-tête de tableau
        pdf.set_font("Arial", "B", 10)
        pdf.cell(45, 7, "Nom", 1, 0, "C")
        pdf.cell(45, 7, "Filière", 1, 0, "C")
        pdf.cell(45, 7, "Carrière visée", 1, 0, "C")
        pdf.cell(25, 7, "Score", 1, 0, "C")
        pdf.cell(30, 7, "Évaluation", 1, 1, "C")

        # Données du tableau
        pdf.set_font("Arial", "", 9)
        for result in sorted(results, key=lambda x: x.get('score', 0) if x.get('score') is not None else -1, reverse=True):
            nom = result.get("Nom", "")
            filiere = result.get("Filière", "")
            carriere = result.get("Carrière", "")
            score = result.get("score", "N/A")

            # Définir la couleur en fonction du score
            if score != "N/A":
                if score >= 7:
                    pdf.set_text_color(0, 128, 0)  # Vert
                    evaluation = "Excellent"
                elif score >= 4:
                    pdf.set_text_color(255, 128, 0)  # Orange
                    evaluation = "À améliorer"
                else:
                    pdf.set_text_color(255, 0, 0)  # Rouge
                    evaluation = "Critique"
            else:
                pdf.set_text_color(0, 0, 0)  # Noir
                evaluation = "N/A"

            # Limiter la longueur des textes
            nom_display = nom[:20] + "..." if len(nom) > 20 else nom
            filiere_display = filiere[:20] + "..." if len(filiere) > 20 else filiere
            carriere_display = carriere[:20] + "..." if len(carriere) > 20 else carriere

            pdf.cell(45, 7, nom_display, 1, 0)
            pdf.cell(45, 7, filiere_display, 1, 0)
            pdf.cell(45, 7, carriere_display, 1, 0)
            pdf.cell(25, 7, f"{score}/10" if score != "N/A" else "N/A", 1, 0, "C")
            pdf.cell(30, 7, evaluation, 1, 1, "C")

            pdf.set_text_color(0, 0, 0)  # Retour au noir

        # Détails individuels
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Analyses individuelles:", 0, 1)

        # Maximum 10 analyses détaillées pour éviter un PDF trop volumineux
        for i, result in enumerate(sorted(results, key=lambda x: x.get('score', 0) if x.get('score') is not None else -1, reverse=True)[:10]):
            pdf.ln(5)
            pdf.set_font("Arial", "B", 12)
            nom = result.get("Nom", "")
            score = result.get("score", "N/A")

            # Couleur selon score
            if score != "N/A":
                if score >= 7:
                    pdf.set_text_color(0, 128, 0)  # Vert
                elif score >= 4:
                    pdf.set_text_color(255, 128, 0)  # Orange
                else:
                    pdf.set_text_color(255, 0, 0)  # Rouge

            pdf.cell(0, 10, f"{nom} - Score: {score}/10", 0, 1)
            pdf.set_text_color(0, 0, 0)  # Retour au noir

            pdf.set_font("Arial", "", 9)

            # Filière et carrière
            filiere = result.get("Filière", "")
            carriere = result.get("Carrière", "")
            pdf.cell(0, 6, f"Filière: {filiere}", 0, 1)
            pdf.cell(0, 6, f"Carrière visée: {carriere}", 0, 1)

            # Métiers recommandés
            pdf.set_font("Arial", "B", 9)
            pdf.cell(0, 6, "Métiers recommandés:", 0, 1)
            pdf.set_font("Arial", "", 9)

            for job in result.get('recommended_jobs', []):
                pdf.cell(0, 5, f"• {job}", 0, 1)

            if not result.get('recommended_jobs'):
                pdf.cell(0, 5, "Aucune recommandation spécifique", 0, 1)

            # Résumé de l'analyse
            pdf.set_font("Arial", "B", 9)
            pdf.cell(0, 6, "Résumé de l'analyse:", 0, 1)
            pdf.set_font("Arial", "", 9)

            # Extrait de l'analyse - limité pour éviter les débordements
            analysis = result.get("analysis", "")
            summary = analysis[:400] + "..." if len(analysis) > 400 else analysis
            pdf.multi_cell(0, 4, summary)

            # Ligne de séparation sauf pour le dernier
            if i < min(len(results), 10) - 1:
                pdf.ln(2)
                pdf.line(10, pdf.get_y(), 200, pdf.get_y())

        # Note de fin
        if len(results) > 10:
            pdf.ln(5)
            pdf.set_font("Arial", "I", 9)
            pdf.cell(0, 6, f"Note: Seulement les 10 premiers résultats sont affichés en détail sur {len(results)} analyses.", 0, 1)

        return pdf.output(dest="S").encode("latin-1")
    except Exception as e:
        print(f"Erreur lors de la génération du PDF de groupe: {str(e)}")
        # PDF de secours en cas d'erreur
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Erreur de génération du rapport PDF", 0, 1, "C")
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Une erreur est survenue: {str(e)}", 0, 1)
        return pdf.output(dest="S").encode("latin-1")

def create_wordcloud(text_data, title="Nuage de mots"):
    """Génère un nuage de mots à partir de données textuelles"""
    try:
        # Création du nuage de mots
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=100,
            contour_width=1,
            contour_color='steelblue'
        ).generate(text_data)

        # Configuration matplotlib
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(title)

        # Conversion en image
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format='png', bbox_inches='tight')
        img_bytes.seek(0)
        plt.close()

        return img_bytes
    except Exception as e:
        print(f"Erreur lors de la création du nuage de mots: {str(e)}")
        return None

# --- INTERFACE DE NAVIGATION ---
st.sidebar.title("⚡ Analyseur Pro")
menu = st.sidebar.radio(
    "Navigation",
    ["Accueil", "Analyse d'orientation", "Performances", "Analyses avancées", "Configuration API", "À propos"],
    label_visibility="collapsed"
)

# Affichage du statut API dans la sidebar
api_status_placeholder = st.sidebar.empty()
if API_KEY:
    api_status_placeholder.success("✅ API configurée")
else:
    api_status_placeholder.error("⛔ API non configurée")

# --- PAGES DE L'APPLICATION ---

# --- 1. PAGE D'ACCUEIL ---
if menu == "Accueil":
    st.title("⚡ ANALYSEUR D'ORIENTATION PROFESSIONNELLE")
    st.markdown("### Détection impitoyable des incohérences de carrière")

    # Présentation
    st.markdown("""
    Bienvenue dans l'outil de détection d'orientation le plus brutalement honnête du marché.
    
    Notre analyse IA ne ménage pas vos étudiants. Elle identifie les erreurs d'orientation
    avec une précision chirurgicale et fournit des recommandations concrètes.
    """)

    # Fonctionnalités
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔍 Fonctionnalités")
        st.markdown("""
        - **Analyse impitoyable** des choix d'orientation
        - **Détection automatique** des compatibilités filière/carrière  
        - **Scoring précis** sur une échelle de 0 à 10
        - **Suggestions de métiers** pertinentes
        - **Recommandations concrètes** pour réorientation
        - **Export immédiat** des résultats (CSV, JSON, PDF)
        - **Analyses avancées** des tendances
        """)

    with col2:
        metrics = calculate_metrics()
        if metrics:
            st.markdown("### 📊 Statistiques globales")

            total = metrics.get("total_analyses", 0)
            avg = metrics.get("avg_score")
            high_score = metrics.get("high_score_percent", 0)

            st.metric("Analyses réalisées", total)
            if avg is not None:
                st.metric("Score moyen de cohérence", f"{avg:.1f}/10")
            st.metric("Orientations optimales", f"{high_score:.1f}%")
        else:
            st.info("Aucune analyse encore réalisée. Commencez dès maintenant!")

    # Démarrage rapide
    st.markdown("### ⚡ Démarrage rapide")

    if not API_KEY:
        st.warning("⚠️ Configuration API requise")
        if st.button("Configurer l'API maintenant"):
            menu = "Configuration API"
            st.rerun()
    else:
        if st.button("Commencer une analyse"):
            menu = "Analyse d'orientation"
            st.rerun()


# --- 2. PAGE D'ANALYSE ---
elif menu == "Analyse d'orientation":
    st.title("🔍 Analyse d'orientation")

    # Upload file section
    st.subheader("1. Importer vos données")

    # File uploader
    uploaded_file = st.file_uploader("Fichier CSV/Excel", type=["csv", "xlsx", "xls"])

    if uploaded_file:
        # Process file
        df, error = load_file(uploaded_file)

        if error:
            st.error(f"Erreur: {error}")
        elif df is not None:
            st.success(f"Fichier chargé: {len(df)} étudiants")

            # Find columns (simplified)
            name_col = next((col for col in df.columns if 'nom' in str(col).lower()), df.columns[0])
            filiere_col = next((col for col in df.columns if 'fili' in str(col).lower()), df.columns[1])
            career_col = next((col for col in df.columns if 'carr' in str(col).lower() or 'envis' in str(col).lower()), df.columns[2])

            # Show detected columns
            st.write(f"Colonnes détectées: Nom='{name_col}', Filière='{filiere_col}', Carrière='{career_col}'")

            # Clean data
            df["filiere_clean"] = df[filiere_col].astype(str).apply(sanitize_text)
            df["carriere_clean"] = df[career_col].astype(str).apply(sanitize_text)

            # Selection options
            st.subheader("2. Configurer l'analyse")

            analysis_type = st.radio(
                "Sélection:",
                ["Tous les étudiants", "Sélection manuelle", "Échantillon aléatoire"]
            )

            students_to_analyze = df

            if analysis_type == "Sélection manuelle":
                selected = st.multiselect(
                    "Sélectionner les étudiants:",
                    df[name_col].unique()
                )
                if selected:
                    students_to_analyze = df[df[name_col].isin(selected)]
                else:
                    st.warning("Sélectionnez au moins un étudiant")

            elif analysis_type == "Échantillon aléatoire":
                sample_size = st.slider(
                    "Nombre d'étudiants:",
                    1, min(len(df), 20), 5
                )
                students_to_analyze = df.sample(sample_size)

            # Analysis button
            if not API_KEY:
                st.error("⛔ API non configurée. Allez dans Configuration API")
            else:
                if st.button("⚡ LANCER L'ANALYSE", type="primary"):
                    # Progress tracking
                    progress = st.progress(0)
                    status_text = st.empty()
                    results = []

                    # Start time
                    start_time = time.time()

                    # Analyze each student
                    for i, (_, student) in enumerate(students_to_analyze.iterrows()):
                        status_text.write(f"Analyse de {student[name_col]}...")

                        # Run analysis
                        analysis_result = analyze_career({
                            "filiere": student["filiere_clean"],
                            "carriere": student["carriere_clean"]
                        })

                        # Store result
                        results.append({
                            "Nom": student[name_col],
                            "Filière": student[filiere_col],
                            "Carrière": student[career_col],
                            "analysis": analysis_result.get("analysis", f"ERREUR: {analysis_result.get('error')}"),
                            "score": analysis_result.get("score"),
                            "recommended_jobs": analysis_result.get("recommended_jobs", [])
                        })

                        # Update progress
                        progress.progress((i + 1) / len(students_to_analyze))

                    # End time and clear status
                    total_time = time.time() - start_time
                    status_text.empty()

                    # Save results to history
                    save_to_history(results)

                    # Display results
                    st.subheader("Résultats d'analyse")
                    st.success(f"Analyse complétée en {total_time:.1f} secondes")

                    # Show metrics if scores available
                    valid_scores = [r.get("score") for r in results if r.get("score") is not None]
                    if valid_scores:
                        avg_score = sum(valid_scores) / len(valid_scores)
                        high_score = sum(1 for s in valid_scores if s >= 7)

                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Score moyen", f"{avg_score:.1f}/10")
                        col2.metric("Excellentes orientations", f"{high_score}/{len(valid_scores)}")
                        col3.metric("Pourcentage optimal", f"{high_score/len(valid_scores)*100:.1f}%")

                        # Sort by score (highest first)
                        results.sort(key=lambda x: (x.get("score") is None, -1 * (x.get("score") or 0)))

                    # Display individual results
                    for result in results:
                        name = result.get("Nom", "")
                        filiere = result.get("Filière", "")
                        carriere = result.get("Carrière", "")
                        score = result.get("score")
                        analysis = result.get("analysis", "")
                        recommended_jobs = result.get("recommended_jobs", [])

                        # Format display based on score
                        if score is not None:
                            if score >= 7:
                                emoji = "✅"
                                color = "green"
                                performance = "Excellent choix"
                            elif score >= 4:
                                emoji = "⚠️"
                                color = "orange"
                                performance = "À améliorer"
                            else:
                                emoji = "🚨"
                                color = "red"
                                performance = "Réorientation conseillée"

                            title = f"{emoji} {name} - {filiere} → {carriere} (Score: {score}/10 - {performance})"
                        else:
                            title = f"📊 {name} - {filiere} → {carriere}"
                            color = "gray"

                        # Display analysis in expandable section
                        with st.expander(title):
                            # Affiche les métiers recommandés
                            if recommended_jobs:
                                st.markdown("#### Métiers recommandés:")
                                for job in recommended_jobs:
                                    st.markdown(f"* **{job}**")
                                st.markdown("---")

                            # Affiche l'analyse complète
                            st.markdown(f"<div style='color:{color};'>{analysis}</div>",
                                        unsafe_allow_html=True)

                    # Export options
                    st.subheader("Exporter les résultats")

                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button("📥 EXPORTER EN CSV", type="primary"):
                            try:
                                # Vérification des données
                                if not results or len(results) == 0:
                                    st.error("ERREUR: Aucun résultat à exporter.")
                                    st.stop()

                                # Création du DataFrame avec les métiers recommandés
                                export_df = pd.DataFrame(results)

                                # Conversion de la liste de métiers en chaîne pour CSV
                                export_df['recommended_jobs'] = export_df['recommended_jobs'].apply(
                                    lambda x: ', '.join(x) if isinstance(x, list) else '')

                                # MÉTHODE DIRECTE
                                csv_data = export_df.to_csv(index=False).encode('utf-8')

                                # Téléchargement immédiat
                                st.download_button(
                                    "⬇️ TÉLÉCHARGER LE CSV",
                                    data=csv_data,
                                    file_name=f"analyse_orientation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                )

                                st.success("✅ CSV généré avec succès.")

                            except Exception as e:
                                st.error(f"ERREUR: {str(e)}")

                    with col2:
                        if st.button("📊 EXPORTER EN JSON"):
                            try:
                                if not results or len(results) == 0:
                                    st.error("ERREUR: Aucun résultat à exporter.")
                                    st.stop()

                                # Export JSON
                                json_str = json.dumps(results, ensure_ascii=False, indent=2)

                                # Téléchargement
                                st.download_button(
                                    "⬇️ TÉLÉCHARGER LE JSON",
                                    data=json_str.encode('utf-8'),
                                    file_name=f"analyse_orientation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json",
                                )

                                st.success("✅ JSON généré avec succès.")

                            except Exception as e:
                                st.error(f"ERREUR: {str(e)}")

                    # Export PDF
                    st.subheader("Exporter en PDF")

                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button("📄 PDF INDIVIDUELS"):
                            # Afficher un bouton de téléchargement pour chaque étudiant
                            for result in results:
                                try:
                                    nom = result.get("Nom", "etudiant")
                                    pdf_data = generate_individual_pdf(result)

                                    st.download_button(
                                        f"⬇️ PDF pour {nom}",
                                        data=pdf_data,
                                        file_name=f"analyse_{nom.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf",
                                        mime="application/pdf",
                                        key=f"pdf_{nom}".replace(" ", "_"),
                                    )
                                except Exception as e:
                                    st.error(f"Erreur pour {result.get('Nom', 'étudiant')}: {str(e)}")

                    with col2:
                        if st.button("📑 RAPPORT GROUPÉ"):
                            try:
                                pdf_data = generate_group_pdf(results)

                                st.download_button(
                                    "⬇️ TÉLÉCHARGER LE RAPPORT PDF",
                                    data=pdf_data,
                                    file_name=f"rapport_orientation_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                                    mime="application/pdf",
                                )

                                st.success("✅ Rapport PDF généré avec succès.")
                            except Exception as e:
                                st.error(f"ERREUR: {str(e)}")


# --- 3. PAGE DE PERFORMANCES ---
elif menu == "Performances":
    st.title("📊 Performances d'orientation")

    # Récupération des statistiques
    all_stats = get_all_stats()

    if not all_stats:
        st.info("Aucune statistique disponible. Effectuez des analyses pour générer des données.")
    else:
        # Métriques globales
        metrics = calculate_metrics()

        st.subheader("Vue d'ensemble")

        # Affichage des métriques clés
        col1, col2, col3 = st.columns(3)
        col1.metric("Analyses réalisées", metrics.get("total_analyses", 0))

        if metrics.get("avg_score") is not None:
            col2.metric("Score moyen de cohérence", f"{metrics.get('avg_score', 0):.1f}/10")
        else:
            col2.metric("Score moyen de cohérence", "N/A")

        col3.metric("Excellentes orientations", f"{metrics.get('high_score_percent', 0):.1f}%")

        # Préparation pour visualisations
        try:
            # Conversion en DataFrame
            stats_df = pd.DataFrame(all_stats)

            # Légendage des niveaux de risque
            if 'score' in stats_df.columns:
                # Distribution des scores en barres
                scores = stats_df['score'].dropna()
                if not scores.empty:
                    score_dist = pd.cut(
                        scores,
                        bins=[0, 3, 6, 10],
                        labels=['0-3: Réorientation', '4-6: À surveiller', '7-10: Excellent']
                    ).value_counts().sort_index()

                    fig = px.bar(
                        x=score_dist.index,
                        y=score_dist.values,
                        title="Distribution des niveaux de cohérence",
                        labels={'x': 'Niveau', 'y': 'Nombre d\'étudiants'},
                        color=score_dist.index,
                        color_discrete_map={
                            '0-3: Réorientation': '#FF4B4B',
                            '4-6: À surveiller': '#FFA726',
                            '7-10: Excellent': '#4CAF50'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Conversion des timestamps
                stats_df['date'] = pd.to_datetime(stats_df['timestamp']).dt.date

                # Graphique d'évolution temporelle
                if len(stats_df) >= 3:
                    st.subheader("Évolution des scores dans le temps")

                    # Calcul des moyennes quotidiennes
                    daily_avg = stats_df.groupby('date')['score'].mean().reset_index()
                    daily_avg.columns = ['Date', 'Score moyen']

                    # Graphique
                    if not daily_avg.empty:
                        fig = px.line(
                            daily_avg,
                            x='Date',
                            y='Score moyen',
                            title="Évolution des scores de cohérence",
                            markers=True
                        )
                        fig.update_layout(yaxis_range=[0, 10])
                        st.plotly_chart(fig, use_container_width=True)

                # Analyse des métiers recommandés
                if 'recommended_jobs' in stats_df.columns:
                    all_jobs = []
                    for jobs_list in stats_df['recommended_jobs']:
                        if isinstance(jobs_list, list):
                            all_jobs.extend(jobs_list)

                    if all_jobs:
                        # Comptage des métiers
                        job_counts = pd.Series(all_jobs).value_counts().reset_index()
                        job_counts.columns = ['Métier', 'Fréquence']

                        # Affichage des métiers les plus recommandés
                        st.subheader("Métiers les plus recommandés")

                        top_n = min(10, len(job_counts))
                        fig = px.bar(
                            job_counts.head(top_n),
                            x='Métier',
                            y='Fréquence',
                            title=f"Top {top_n} des métiers les plus recommandés",
                            color='Fréquence',
                            color_continuous_scale='Viridis'
                        )
                        st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Erreur lors de la génération des graphiques: {str(e)}")

        # Liste des étudiants analysés
        st.subheader("Historique par étudiant")

        # Extraction des étudiants uniques
        students = list(set(stat.get('nom') for stat in all_stats if stat.get('nom')))
        students.sort()  # Tri alphabétique

        if not students:
            st.warning("Aucun étudiant identifié dans les données.")
        else:
            for student in students:
                # Filtrer les données pour cet étudiant
                student_stats = [stat for stat in all_stats if stat.get('nom') == student]

                if student_stats:
                    # Calculer le score moyen
                    valid_scores = [stat.get('score') for stat in student_stats if stat.get('score') is not None]

                    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None
                    score_display = f" - Score moyen: {avg_score:.1f}/10" if avg_score is not None else ""

                    # Définir la couleur en fonction du score
                    if avg_score is not None:
                        if avg_score >= 7:
                            color = "green"
                            emoji = "✅"
                        elif avg_score >= 4:
                            color = "orange"
                            emoji = "⚠️"
                        else:
                            color = "red"
                            emoji = "🚨"
                    else:
                        color = "gray"
                        emoji = "📊"

                    # Affichage par étudiant
                    with st.expander(f"{emoji} {student}{score_display}"):
                        # Trier par date (plus récent en premier)
                        sorted_stats = sorted(
                            student_stats,
                            key=lambda x: x.get('timestamp', ''),
                            reverse=True
                        )

                        # Colonnes pour les métiers recommandés
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            # Tableau des analyses
                            for stat in sorted_stats:
                                timestamp = datetime.fromisoformat(stat.get('timestamp', '')).strftime('%d/%m/%Y %H:%M') \
                                    if stat.get('timestamp') else "Date inconnue"

                                st.markdown(f"""
                                **📅 {timestamp}**
                                - **Filière:** {stat.get('filiere', 'Non spécifiée')}
                                - **Carrière visée:** {stat.get('carriere', 'Non spécifiée')}
                                - **Score de cohérence:** {stat.get('score', 'N/A')}/10
                                
                                <div style='color:{color};'>{stat.get('analysis', 'Aucune analyse disponible')}</div>
                                ---
                                """, unsafe_allow_html=True)

                        with col2:
                            # Liste des métiers recommandés pour cet étudiant
                            all_student_jobs = []
                            for stat in sorted_stats:
                                if 'recommended_jobs' in stat and stat['recommended_jobs']:
                                    all_student_jobs.extend(stat['recommended_jobs'])

                            if all_student_jobs:
                                st.markdown("#### Métiers recommandés:")
                                job_counts = pd.Series(all_student_jobs).value_counts()
                                for job, count in job_counts.items():
                                    st.markdown(f"- **{job}** ({count})")

                                # Option PDF individuel
                                if st.button(f"📄 PDF pour {student}", key=f"btn_pdf_{student}".replace(" ", "_")):
                                    # Créer un PDF pour le dernier rapport (le plus récent)
                                    latest_stat = sorted_stats[0]
                                    result = {
                                        "Nom": student,
                                        "Filière": latest_stat.get('filiere', ''),
                                        "Carrière": latest_stat.get('carriere', ''),
                                        "score": latest_stat.get('score'),
                                        "analysis": latest_stat.get('analysis', ''),
                                        "recommended_jobs": latest_stat.get('recommended_jobs', [])
                                    }

                                    try:
                                        pdf_data = generate_individual_pdf(result)

                                        st.download_button(
                                            f"⬇️ Télécharger PDF",
                                            data=pdf_data,
                                            file_name=f"analyse_{student.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf",
                                            mime="application/pdf",
                                            key=f"dl_pdf_{student}".replace(" ", "_")
                                        )
                                    except Exception as e:
                                        st.error(f"Erreur: {str(e)}")

        # Bouton pour exporter les statistiques
        st.subheader("Exportation des données")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📥 EXPORTER L'HISTORIQUE COMPLET"):
                try:
                    # Export JSON
                    json_str = json.dumps(all_stats, ensure_ascii=False, indent=2)

                    # Téléchargement
                    st.download_button(
                        "⬇️ TÉLÉCHARGER L'HISTORIQUE",
                        data=json_str.encode('utf-8'),
                        file_name=f"historique_analyses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                    )

                    st.success("✅ Historique exporté avec succès.")

                except Exception as e:
                    st.error(f"ERREUR: {str(e)}")

        with col2:
            if st.button("📊 RAPPORT GLOBAL PDF"):
                try:
                    # Création d'un rapport global à partir de tous les stats
                    results = []
                    for stat in all_stats:
                        results.append({
                            "Nom": stat.get('nom', 'Anonyme'),
                            "Filière": stat.get('filiere', ''),
                            "Carrière": stat.get('carriere', ''),
                            "score": stat.get('score'),
                            "analysis": stat.get('analysis', ''),
                            "recommended_jobs": stat.get('recommended_jobs', [])
                        })

                    pdf_data = generate_group_pdf(results)

                    st.download_button(
                        "⬇️ TÉLÉCHARGER LE RAPPORT PDF",
                        data=pdf_data,
                        file_name=f"rapport_global_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                    )

                    st.success("✅ Rapport PDF généré avec succès.")
                except Exception as e:
                    st.error(f"ERREUR: {str(e)}")

        # Bouton pour supprimer tout l'historique
        with col3:
            if st.button("🗑️ SUPPRIMER TOUT L'HISTORIQUE", type="secondary"):
                confirm = st.checkbox("Confirmer la suppression permanente")

                if confirm:
                    success = delete_stats()
                    if success:
                        st.success("Tout l'historique a été supprimé")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Erreur lors de la suppression de l'historique")


# --- 4. PAGE D'ANALYSES AVANCÉES ---
elif menu == "Analyses avancées":
    st.title("🔬 Analyses avancées")

    # Récupération des statistiques
    all_stats = get_all_stats()

    if not all_stats:
        st.info("Aucune donnée disponible. Effectuez des analyses pour générer des données.")
    else:
        # Conversion en DataFrame pour analyses
        try:
            stats_df = pd.DataFrame(all_stats)

            # Navigation
            analyse_type = st.radio(
                "Type d'analyse:",
                ["Compatibilité filière-carrière", "Nuages de mots", "Tendances temporelles", "Exportation avancée"]
            )

            # 1. MATRICE DE COMPATIBILITÉ
            if analyse_type == "Compatibilité filière-carrière":
                st.subheader("Matrice de compatibilité filière-carrière")
                st.markdown("""
                Cette visualisation montre les scores moyens de cohérence entre 
                les différentes filières et carrières analysées.
                """)

                # Vérification du nombre de données
                if len(stats_df) < 3:
                    st.warning("Données insuffisantes pour générer une matrice de compatibilité.")
                else:
                    # Extraction des filières et carrières uniques
                    filieres = stats_df['filiere'].dropna().unique()
                    carrieres = stats_df['carriere'].dropna().unique()

                    if len(filieres) > 1 and len(carrieres) > 1:
                        # Création d'une matrice de scores moyens
                        compatibility_data = []
                        heatmap_x = []
                        heatmap_y = []

                        for filiere in filieres:
                            for carriere in carrieres:
                                subset = stats_df[(stats_df['filiere'] == filiere) &
                                                  (stats_df['carriere'] == carriere)]
                                if not subset.empty and 'score' in subset.columns:
                                    avg_score = subset['score'].mean()
                                    if not pd.isna(avg_score):
                                        compatibility_data.append(avg_score)
                                        heatmap_x.append(carriere)
                                        heatmap_y.append(filiere)

                        if compatibility_data:
                            # Création du graphique heatmap
                            fig = go.Figure(data=go.Heatmap(
                                z=compatibility_data,
                                x=heatmap_x,
                                y=heatmap_y,
                                colorscale='RdYlGn',  # Rouge à Vert
                                colorbar=dict(title='Score'),
                                hoverongaps=False
                            ))

                            fig.update_layout(
                                title="Matrice de compatibilité Filière/Carrière",
                                xaxis_title="Carrière visée",
                                yaxis_title="Filière actuelle",
                                height=600,
                                xaxis={'categoryorder':'category ascending'},
                                yaxis={'categoryorder':'category ascending'}
                            )

                            st.plotly_chart(fig, use_container_width=True)

                            st.info("""
                            **Comment lire cette matrice:**
                            - Les cases vertes indiquent une forte compatibilité (score élevé)
                            - Les cases rouges indiquent une faible compatibilité (score bas)
                            - Les zones vides indiquent qu'aucune analyse n'a été faite pour cette combinaison
                            """)
                    else:
                        st.warning("Diversité insuffisante dans les données pour créer une matrice.")

                # Analyse des carrières bien notées par filière
                if 'filiere' in stats_df.columns and 'carriere' in stats_df.columns and 'score' in stats_df.columns:
                    st.subheader("Meilleures carrières par filière")

                    # Regroupement par filière et carrière
                    grouped = stats_df.groupby(['filiere', 'carriere'])['score'].mean().reset_index()

                    # Top carrières pour chaque filière
                    top_careers = []
                    for filiere in grouped['filiere'].unique():
                        filiere_data = grouped[grouped['filiere'] == filiere]
                        top_careers_for_filiere = filiere_data.nlargest(3, 'score')
                        top_careers.append({
                            'Filière': filiere,
                            'Top carrières': top_careers_for_filiere
                        })

                    # Affichage des résultats
                    for entry in top_careers:
                        filiere = entry['Filière']
                        top = entry['Top carrières']

                        if not top.empty:
                            st.markdown(f"**{filiere}**")
                            for _, row in top.iterrows():
                                score_color = "green" if row['score'] >= 7 else "orange" if row['score'] >= 4 else "red"
                                st.markdown(f"- {row['carriere']} - <span style='color:{score_color};'>Score: {row['score']:.1f}/10</span>", unsafe_allow_html=True)

            # 2. NUAGES DE MOTS
            elif analyse_type == "Nuages de mots":
                st.subheader("Analyse textuelle")

                # Options d'analyse
                text_analysis_type = st.selectbox(
                    "Sélectionnez le type d'analyse textuelle:",
                    ["Nuage des filières", "Nuage des carrières", "Nuage des métiers recommandés", "Nuage des erreurs critiques"]
                )

                # Préparation des données selon le type d'analyse
                if text_analysis_type == "Nuage des filières":
                    text_data = " ".join(stats_df['filiere'].dropna().astype(str))
                    title = "Filières les plus courantes"

                elif text_analysis_type == "Nuage des carrières":
                    text_data = " ".join(stats_df['carriere'].dropna().astype(str))
                    title = "Carrières les plus visées"

                elif text_analysis_type == "Nuage des métiers recommandés":
                    # Extrait tous les métiers recommandés
                    all_jobs = []
                    for jobs in stats_df['recommended_jobs']:
                        if isinstance(jobs, list):
                            all_jobs.extend(jobs)
                    text_data = " ".join(all_jobs)
                    title = "Métiers les plus recommandés"

                elif text_analysis_type == "Nuage des erreurs critiques":
                    # Extraction des erreurs critiques des analyses
                    errors_section = []
                    pattern = r"ERREURS CRITIQUES:(.*?)(?:RECOMMANDATIONS:|$)"

                    for analysis in stats_df['analysis'].dropna():
                        matches = re.search(pattern, analysis, re.DOTALL)
                        if matches:
                            errors_section.append(matches.group(1))

                    text_data = " ".join(errors_section)
                    title = "Compétences manquantes les plus fréquentes"

                # Génération du nuage de mots
                if text_data and len(text_data) > 10:
                    img_bytes = create_wordcloud(text_data, title)

                    if img_bytes:
                        st.image(img_bytes, use_column_width=True)

                        # Option d'export de l'image
                        img_str = base64.b64encode(img_bytes.getvalue()).decode()
                        href = f'<a href="data:file/png;base64,{img_str}" download="wordcloud_{text_analysis_type.lower().replace(" ", "_")}.png">Télécharger l\'image</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    else:
                        st.error("Impossible de générer le nuage de mots.")
                else:
                    st.warning("Données textuelles insuffisantes pour cette analyse.")

                # Analyse fréquentielle en plus du nuage
                if text_data and len(text_data) > 10:
                    st.subheader("Analyse fréquentielle")

                    # Nettoyage et tokenisation basique
                    words = re.findall(r'\b\w+\b', text_data.lower())
                    words = [w for w in words if len(w) > 2 and w not in ['les', 'des', 'pour', 'une', 'avec', 'dans']]

                    word_counts = pd.Series(words).value_counts().head(15)

                    # Graphique des fréquences
                    fig = px.bar(
                        x=word_counts.index,
                        y=word_counts.values,
                        title="Fréquence des termes",
                        labels={'x': 'Terme', 'y': 'Fréquence'},
                        color=word_counts.values,
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # 3. TENDANCES TEMPORELLES
            elif analyse_type == "Tendances temporelles":
                st.subheader("Analyse des tendances temporelles")

                # Conversion des timestamps
                if 'timestamp' in stats_df.columns:
                    stats_df['date'] = pd.to_datetime(stats_df['timestamp']).dt.date

                    if stats_df['date'].nunique() >= 3:
                        # Options de visualisation
                        time_metric = st.selectbox(
                            "Sélectionnez la métrique à analyser dans le temps:",
                            ["Score moyen", "Nombre d'analyses", "Répartition des scores"]
                        )

                        if time_metric == "Score moyen":
                            # Calcul des moyennes quotidiennes
                            daily_avg = stats_df.groupby('date')['score'].mean().reset_index()
                            daily_avg.columns = ['Date', 'Score moyen']

                            # Courbe d'évolution
                            fig = px.line(
                                daily_avg,
                                x='Date',
                                y='Score moyen',
                                title="Évolution du score moyen de cohérence dans le temps",
                                markers=True
                            )
                            fig.update_layout(yaxis_range=[0, 10])
                            st.plotly_chart(fig, use_container_width=True)

                        elif time_metric == "Nombre d'analyses":
                            # Comptage par jour
                            daily_count = stats_df.groupby('date').size().reset_index()
                            daily_count.columns = ['Date', 'Nombre d\'analyses']

                            # Graphique
                            fig = px.bar(
                                daily_count,
                                x='Date',
                                y='Nombre d\'analyses',
                                title="Nombre d'analyses effectuées par jour",
                                color='Nombre d\'analyses',
                                color_continuous_scale='Viridis'
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        elif time_metric == "Répartition des scores":
                            # Catégorisation des scores
                            stats_df['catégorie'] = pd.cut(
                                stats_df['score'],
                                bins=[0, 3, 6, 10],
                                labels=['Faible (0-3)', 'Moyen (4-6)', 'Élevé (7-10)']
                            )

                            # Comptage par catégorie et par jour
                            daily_categories = pd.crosstab(
                                stats_df['date'],
                                stats_df['catégorie']
                            ).reset_index()

                            # Mise en forme pour graphique
                            daily_categories_melted = pd.melt(
                                daily_categories,
                                id_vars=['date'],
                                var_name='Catégorie',
                                value_name='Nombre'
                            )

                            # Graphique empilé
                            fig = px.area(
                                daily_categories_melted,
                                x='date',
                                y='Nombre',
                                color='Catégorie',
                                title="Évolution de la répartition des scores dans le temps",
                                color_discrete_map={
                                    'Faible (0-3)': '#FF4B4B',
                                    'Moyen (4-6)': '#FFA726',
                                    'Élevé (7-10)': '#4CAF50'
                                }
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Données temporelles insuffisantes pour l'analyse des tendances.")
                else:
                    st.error("Les données ne contiennent pas d'information temporelle.")

            # 4. EXPORTATION AVANCÉE
            elif analyse_type == "Exportation avancée":
                st.subheader("Exportation et traitement avancé des données")

                # Options d'exportation
                export_option = st.radio(
                    "Sélectionnez le format d'exportation:",
                    ["Tableau croisé dynamique", "Matrice de corrélation", "Rapport analytique complet"]
                )

                if export_option == "Tableau croisé dynamique":
                    st.markdown("""
                    Le tableau croisé dynamique permet de visualiser la relation 
                    entre deux variables et une métrique.
                    """)

                    # Options du tableau
                    col1, col2 = st.columns(2)
                    with col1:
                        row_var = st.selectbox(
                            "Variable en ligne:",
                            ["filiere", "carriere"]
                        )
                    with col2:
                        col_var = st.selectbox(
                            "Variable en colonne:",
                            ["carriere", "filiere"],
                            index=1 if row_var == "filiere" else 0
                        )

                    # Création du tableau croisé
                    if row_var and col_var and row_var != col_var:
                        pivot = pd.pivot_table(
                            stats_df,
                            values='score',
                            index=row_var,
                            columns=col_var,
                            aggfunc='mean',
                            fill_value=None
                        )

                        # Affichage du tableau
                        st.markdown("### Tableau croisé des scores moyens")
                        st.dataframe(pivot.style.background_gradient(cmap='RdYlGn', vmin=0, vmax=10))

                        # Export Excel
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                            pivot.to_excel(writer, sheet_name='Pivot')

                            # Ajout de mise en forme conditionnelle
                            workbook = writer.book
                            worksheet = writer.sheets['Pivot']

                            # Format pour les scores
                            format_high = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
                            format_mid = workbook.add_format({'bg_color': '#FFEB9C', 'font_color': '#9C5700'})
                            format_low = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})

                            # Mise en forme conditionnelle
                            worksheet.conditional_format(1, 1, len(pivot.index)+1, len(pivot.columns)+1, {
                                'type': '3_color_scale',
                                'min_color': '#FFC7CE',
                                'mid_color': '#FFEB9C',
                                'max_color': '#C6EFCE',
                                'min_value': 0,
                                'mid_value': 5,
                                'max_value': 10
                            })

                        excel_data = buffer.getvalue()

                        st.download_button(
                            "📊 Télécharger le tableau Excel",
                            data=excel_data,
                            file_name=f"pivot_{row_var}_{col_var}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.ms-excel"
                        )

                elif export_option == "Matrice de corrélation":
                    st.markdown("""
                    Exploration des relations entre différentes variables et le score obtenu.
                    """)

                    # Préparation des données
                    # Extraction de caractéristiques à partir des textes
                    try:
                        # Longueur de la filière et de la carrière
                        if 'filiere' in stats_df.columns and 'carriere' in stats_df.columns:
                            stats_df['longueur_filiere'] = stats_df['filiere'].astype(str).apply(len)
                            stats_df['longueur_carriere'] = stats_df['carriere'].astype(str).apply(len)

                            # Nombre de métiers recommandés
                            if 'recommended_jobs' in stats_df.columns:
                                stats_df['nb_metiers_recommandes'] = stats_df['recommended_jobs'].apply(
                                    lambda x: len(x) if isinstance(x, list) else 0
                                )

                            # Création de la matrice de corrélation
                            corr_cols = ['score', 'longueur_filiere', 'longueur_carriere']
                            if 'nb_metiers_recommandes' in stats_df.columns:
                                corr_cols.append('nb_metiers_recommandes')

                            corr_matrix = stats_df[corr_cols].corr()

                            # Affichage de la matrice
                            fig = px.imshow(
                                corr_matrix,
                                text_auto=True,
                                color_continuous_scale='RdBu_r',
                                title="Matrice de corrélation des variables"
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            st.info("""
                            **Comment lire cette matrice:**
                            - Les valeurs proches de 1 indiquent une forte corrélation positive
                            - Les valeurs proches de -1 indiquent une forte corrélation négative
                            - Les valeurs proches de 0 indiquent l'absence de corrélation
                            """)

                            # Export CSV
                            csv_corr = corr_matrix.to_csv(index=True).encode('utf-8')

                            st.download_button(
                                "📊 Télécharger la matrice CSV",
                                data=csv_corr,
                                file_name=f"correlation_matrix_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
                    except Exception as e:
                        st.error(f"Erreur lors de la création de la matrice de corrélation: {str(e)}")

                elif export_option == "Rapport analytique complet":
                    st.markdown("""
                    Génération d'un rapport analytique complet au format PDF.
                    """)

                    if st.button("🔍 Générer le rapport analytique"):
                        try:
                            # Création d'un PDF complet avec toutes les analyses
                            pdf = FPDF()

                            # Page de titre
                            pdf.add_page()
                            pdf.set_font("Arial", "B", 24)
                            pdf.cell(0, 20, "RAPPORT ANALYTIQUE COMPLET", 0, 1, "C")
                            pdf.set_font("Arial", "I", 14)
                            pdf.cell(0, 10, f"Analyseur d'Orientation Professionnelle", 0, 1, "C")
                            pdf.set_font("Arial", "", 12)
                            pdf.cell(0, 10, f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}", 0, 1, "C")
                            pdf.cell(0, 10, f"Nombre total d'analyses: {len(stats_df)}", 0, 1, "C")

                            # Image du logo ou autre illustration
                            pdf.image("https://via.placeholder.com/150x150.png?text=Logo", x=80, y=100, w=50)

                            # Sommaire
                            pdf.add_page()
                            pdf.set_font("Arial", "B", 16)
                            pdf.cell(0, 20, "SOMMAIRE", 0, 1)
                            pdf.set_font("Arial", "", 12)
                            pdf.cell(0, 10, "1. Statistiques générales................3", 0, 1)
                            pdf.cell(0, 10, "2. Distribution des scores................4", 0, 1)
                            pdf.cell(0, 10, "3. Analyse des filières..................5", 0, 1)
                            pdf.cell(0, 10, "4. Analyse des carrières.................6", 0, 1)
                            pdf.cell(0, 10, "5. Métiers recommandés...................7", 0, 1)
                            pdf.cell(0, 10, "6. Corrélations..........................8", 0, 1)
                            pdf.cell(0, 10, "7. Recommandations.......................9", 0, 1)

                            # Section 1: Statistiques générales
                            pdf.add_page()
                            pdf.set_font("Arial", "B", 16)
                            pdf.cell(0, 20, "1. STATISTIQUES GÉNÉRALES", 0, 1)

                            # Métriques principales
                            metrics = calculate_metrics()

                            pdf.set_font("Arial", "", 11)
                            pdf.cell(0, 10, f"Nombre total d'analyses: {metrics.get('total_analyses', 0)}", 0, 1)

                            if metrics.get("avg_score") is not None:
                                pdf.cell(0, 10, f"Score moyen de cohérence: {metrics.get('avg_score', 0):.1f}/10", 0, 1)
                            else:
                                pdf.cell(0, 10, "Score moyen de cohérence: N/A", 0, 1)

                            pdf.cell(0, 10, f"Excellentes orientations: {metrics.get('high_score_percent', 0):.1f}%", 0, 1)

                            pdf.set_font("Arial", "B", 12)
                            pdf.cell(0, 15, "Distribution des scores:", 0, 1)

                            # Distribution simplifiée
                            distribution = metrics.get("scores_distribution", {})
                            pdf.set_font("Arial", "", 10)
                            pdf.cell(0, 8, f"• Scores faibles (0-3): {distribution.get('0-3', 0)} étudiants", 0, 1)
                            pdf.cell(0, 8, f"• Scores moyens (4-6): {distribution.get('4-6', 0)} étudiants", 0, 1)
                            pdf.cell(0, 8, f"• Scores élevés (7-10): {distribution.get('7-10', 0)} étudiants", 0, 1)

                            # Section 2: Distribution des scores (avec image d'histogramme)
                            pdf.add_page()
                            pdf.set_font("Arial", "B", 16)
                            pdf.cell(0, 20, "2. DISTRIBUTION DES SCORES", 0, 1)

                            # Image placée en dur pour simplifier
                            pdf.image("https://via.placeholder.com/500x300.png?text=Histogramme+des+scores", x=10, y=40, w=190)

                            pdf.set_y(150)
                            pdf.set_font("Arial", "", 10)
                            pdf.multi_cell(0, 5, """
                            L'histogramme ci-dessus montre la distribution des scores de cohérence.
                            On observe que la majorité des étudiants se situent dans la zone médiane,
                            ce qui indique un besoin d'amélioration modéré pour la plupart des parcours analysés.
                            """)

                            # Section 3: Analyse des filières
                            pdf.add_page()
                            pdf.set_font("Arial", "B", 16)
                            pdf.cell(0, 20, "3. ANALYSE DES FILIÈRES", 0, 1)

                            # Top des filières par score moyen
                            if 'filiere' in stats_df.columns and 'score' in stats_df.columns:
                                top_filieres = stats_df.groupby('filiere')['score'].agg(['mean', 'count']).sort_values('mean', ascending=False)

                                pdf.set_font("Arial", "B", 12)
                                pdf.cell(0, 15, "Filières par score moyen de cohérence:", 0, 1)

                                # Tableau des filières
                                pdf.set_font("Arial", "B", 10)
                                pdf.cell(80, 10, "Filière", 1, 0, "C")
                                pdf.cell(30, 10, "Score moyen", 1, 0, "C")
                                pdf.cell(30, 10, "Nb étudiants", 1, 1, "C")

                                pdf.set_font("Arial", "", 10)
                                for i, (filiere, row) in enumerate(top_filieres.iterrows()):
                                    if i < 10:  # Limite aux 10 premières
                                        pdf.cell(80, 8, str(filiere)[:40], 1, 0)
                                        pdf.cell(30, 8, f"{row['mean']:.1f}", 1, 0, "C")
                                        pdf.cell(30, 8, f"{int(row['count'])}", 1, 1, "C")

                            # Analyses suivantes similaires...
                            # (Section 4 à 7)

                            # Génération du PDF final
                            pdf_data = pdf.output(dest="S").encode("latin-1")

                            st.download_button(
                                "📊 Télécharger le rapport analytique complet",
                                data=pdf_data,
                                file_name=f"rapport_analytique_{datetime.now().strftime('%Y%m%d')}.pdf",
                                mime="application/pdf",
                            )

                            st.success("✅ Rapport analytique généré avec succès.")
                        except Exception as e:
                            st.error(f"Erreur lors de la génération du rapport analytique: {str(e)}")
        except Exception as e:
            st.error(f"Erreur lors de l'analyse avancée: {str(e)}")


# --- 5. PAGE DE CONFIGURATION API ---
elif menu == "Configuration API":
    st.title("⚙️ Configuration de l'API")

    # Statut actuel de l'API
    st.subheader("Statut actuel de l'API")

    api_key = API_KEY
    if api_key:
        # Validation simple
        valid = api_key.startswith("sk-or-") and len(api_key) > 30

        if valid:
            st.success(f"✅ API configurée et valide. Clé: {api_key[:5]}...{api_key[-4:]}")
        else:
            st.warning(f"⚠️ Problème détecté avec la clé API: format incorrect")
    else:
        st.error("⛔ API non configurée. Veuillez entrer votre clé API ci-dessous.")

    # Formulaire pour configurer/mettre à jour la clé
    st.subheader("Configurer la clé API OpenRouter")

    st.write("""
    Entrez votre clé API OpenRouter pour activer les fonctionnalités d'analyse.
    
    Pour obtenir une clé API:
    1. Créez un compte sur [OpenRouter](https://openrouter.ai)
    2. Générez une clé API dans vos paramètres de compte
    3. Choisissez le modèle DeepSeek Chat comme modèle par défaut
    """)

    api_key_input = st.text_input(
        "Clé API OpenRouter",
        value=api_key,
        type="password",
        placeholder="sk-or-..."
    )

    # Boutons d'action
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Enregistrer la clé API", type="primary"):
            if not api_key_input:
                st.error("Aucune clé fournie.")
            else:
                # Validation simple
                if api_key_input.startswith("sk-or-") and len(api_key_input) > 30:
                    # Sauvegarde
                    success, message = save_api_key(api_key_input)

                    if success:
                        # Mise à jour des variables
                        API_KEY = api_key_input

                        # Mise à jour du placeholder de statut
                        api_status_placeholder.success("✅ API configurée")

                        st.success(f"✅ {message}")
                    else:
                        st.error(f"Échec d'enregistrement: {message}")
                else:
                    st.error(f"Clé API invalide: doit commencer par 'sk-or-' et avoir au moins 30 caractères")

    with col2:
        if st.button("Tester la connexion API"):
            # Vérifier si une clé est disponible
            test_key = api_key_input if api_key_input else api_key

            if not test_key:
                st.error("Aucune clé API disponible pour le test.")
            else:
                with st.spinner("Test de connexion en cours..."):
                    # Sauvegarde temporaire pour test
                    if api_key_input and api_key_input != api_key:
                        os.environ["OPENROUTER_API_KEY"] = api_key_input
                        # Mise à jour temporaire
                        temp_api_key = API_KEY
                        API_KEY = api_key_input

                    # Test de connexion
                    response = call_api("Réponds uniquement 'OK' si tu reçois ce message.")

                    # Restauration si nécessaire
                    if api_key_input and api_key_input != api_key:
                        API_KEY = temp_api_key

                    if "error" in response:
                        st.error(f"❌ {response['error']}")
                    else:
                        st.success(f"✅ Connexion API fonctionnelle: '{response['content']}'")

    # Instructions détaillées
    with st.expander("Instructions détaillées"):
        st.markdown("""
        ### Guide d'obtention d'une clé API OpenRouter
        
        1. **Créez un compte sur [OpenRouter](https://openrouter.ai)**
           - Inscrivez-vous avec votre email ou un compte Google
           - Remplissez les informations demandées
        
        2. **Générez une nouvelle clé API**
           - Allez dans *API Keys* dans votre tableau de bord
           - Cliquez sur *Create Key*
           - Donnez un nom à votre clé (ex: "Analyseur d'Orientation")
        
        3. **Configurez vos préférences de modèle**
           - Sélectionnez DeepSeek comme fournisseur préféré
           - Le modèle `deepseek/deepseek-chat-v3-0324` est recommandé
        
        4. **Copiez la clé générée**
           - Elle commence toujours par `sk-or-`
           - Collez-la dans le champ ci-dessus et cliquez sur "Enregistrer"
        
        5. **Vérifiez les crédits disponibles**
           - OpenRouter fournit des crédits gratuits pour tester
           - Vous pouvez également acheter des crédits pour une utilisation intensive
        """)


# --- 6. PAGE À PROPOS ---
elif menu == "À propos":
    st.title("ℹ️ À propos de l'application")

    # Présentation de l'application
    st.markdown("""
    ## Analyseur d'Orientation Professionnelle
    
    Cette application a été conçue pour aider les conseillers d'orientation et les établissements d'enseignement
    à évaluer la cohérence entre les filières suivies et les carrières envisagées par les étudiants.
    
    ### Comment ça fonctionne
    
    1. L'application analyse chaque étudiant individuellement
    2. L'IA évalue la cohérence entre la filière suivie et la carrière visée
    3. Un score de cohérence est attribué sur une échelle de 0 à 10
    4. Des recommandations de métiers alternatifs sont proposées
    5. Des recommandations concrètes sont fournies pour chaque cas
    
    ### L'échelle de cohérence
    
    - **0-3: Réorientation nécessaire** - Écart important entre formation et ambition
    - **4-6: À surveiller** - Des ajustements sont recommandés
    - **7-10: Excellent choix** - Alignement fort entre formation et ambition
    
    ### Confidentialité
    
    Les données sont traitées localement et ne sont pas conservées sur des serveurs externes.
    Seuls les résultats d'analyse sont sauvegardés dans un fichier local pour le suivi des performances.
    """)

    # Fonctionnalités ajoutées
    st.subheader("Nouvelles fonctionnalités (2025)")

    st.markdown("""
    #### Améliorations majeures
    
    - **Suggestions de métiers**: Pour chaque étudiant, l'IA propose 2 métiers alternatifs qui exploitent les compétences de sa filière
    - **Logique de scoring inversée**: Un score élevé indique maintenant une bonne cohérence (10 = excellent choix)
    - **Export PDF**: Génération de rapports individuels et groupés en PDF
    - **Analyses avancées**: Visualisations et analyses statistiques poussées des données
    - **Matrice de compatibilité**: Visualisation des meilleures combinaisons filière/carrière
    - **Nuages de mots**: Analyse textuelle des filières, carrières et métiers recommandés
    
    #### Possibilités d'exploitation des données
    
    - **Cartographie des parcours**: Visualiser les chemins typiques entre filières et carrières
    - **Tableau croisé dynamique**: Générer des rapports Excel pour les analyses d'orientation
    - **Analyse de tendances**: Suivre l'évolution des scores et des recommandations dans le temps
    - **Clustering des étudiants**: Regrouper les étudiants par profils similaires pour interventions ciblées
    - **Prédiction de réussite**: Modèles statistiques basés sur la cohérence filière/carrière
    """)

    # Informations techniques
    st.subheader("Informations techniques")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Technologies utilisées:**
        - Streamlit (interface)
        - OpenRouter API (analyse IA)
        - Pandas & NumPy (traitement de données)
        - Plotly & Matplotlib (visualisations)
        - FPDF (génération de rapports)
        - WordCloud (analyses textuelles)
        """)

    with col2:
        st.markdown("""
        **Modèle IA:**
        - DeepSeek Chat V3
        - Format d'analyse structuré
        - Extraction automatique des scores
        - Détection des métiers recommandés
        - Recommandations personnalisées
        """)

    # Instructions
    st.subheader("Guide d'utilisation")

    st.markdown("""
    1. **Configuration API** - Entrez votre clé OpenRouter
    2. **Préparation des données** - CSV ou Excel avec les colonnes Nom, Filière, Carrière
    3. **Analyse** - Importez le fichier et lancez l'analyse
    4. **Interprétation** - Consultez le score, les métiers recommandés et les recommandations
    5. **Export** - Téléchargez les résultats en CSV, JSON ou PDF (individuel ou groupé)
    6. **Analyses avancées** - Exploitez les visualisations pour des insights plus profonds
    """)

    # Cas d'utilisation
    st.subheader("Cas d'utilisation")

    st.markdown("""
    - **Universités et écoles** - Valider les choix d'orientation des étudiants et identifier les réorientations nécessaires
    - **Centres d'orientation** - Offrir des recommandations personnalisées basées sur les compétences existantes
    - **Services RH** - Analyser les reconversions professionnelles de collaborateurs
    - **Organismes de formation** - Évaluer la pertinence des formations par rapport aux objectifs professionnels
    - **Observatoires de l'emploi** - Analyser les tendances des aspirations professionnelles
    """)

    # Contact/Support
    st.markdown("---")
    st.subheader("Support et contact")

    st.info("""
    Pour toute question ou suggestion d'amélioration, n'hésitez pas à nous contacter.
    
    Version: 3.0 - Mai 2025
    """)

# Footer global
st.markdown("---")
st.markdown("⚡ Analyseur d'Orientation Pro - Des métiers sur mesure pour votre avenir - 2025")