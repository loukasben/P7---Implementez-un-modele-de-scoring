# Projet 7 : Implémentez un Modèle de Scoring
# Le projet permet d'évaluer le risque de non-remboursement d'un crédit en fonction des données client.
# Ce fichier contient les liens de l'API et du Dashboard déployés dans le cloud.
# Exécutez ce script en ligne de commande pour afficher les adresses de l'API et du Dashboard :
# python nom_du_fichier.py

API_URL = "https://p7-implementez-un-modele-de-scoring.onrender.com"
DASHBOARD_URL = "https://p7-dashboard.onrender.com"

# Affiche le Dashboard
def display_dashboard():
    print("Adresses de déploiement :")
    print(f"API : {API_URL}")
    print(f"Dashboard : {DASHBOARD_URL}")

# Exécute la fonction si le script est exécuté directement
if __name__ == "__main__":
    display_dashboard()
