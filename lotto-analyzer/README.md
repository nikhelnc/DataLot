# Lotto Analyzer

**Statistical analysis and probability modeling for lottery draws with scientific rigor**

A comprehensive application for analyzing lottery draw data with emphasis on:
- Statistical testing (uniformity, independence, randomness)
- Anomaly detection (drift, change points, outliers)
- Probability modeling with walk-forward evaluation
- Responsible interpretation with mandatory limitations disclosure

## 🎯 Key Features

### Data Management
- **CSV Import**: Preview and commit workflow with validation
- **Multi-game support**: Configurable rules per game (numbers range, bonus, calendar)
- **Data quality checks**: Validation, duplicate detection, audit trail

### Statistical Analysis
- **Descriptive metrics**: Frequencies, entropy, KL divergence, PSI
- **Randomness tests**: χ² uniformity, runs tests, independence checks
- **FDR correction**: Benjamini-Hochberg multi-test correction
- **Meta-tests**: P-value distribution analysis

### Anomaly Detection
- **Drift detection**: PSI and KL divergence with Monte Carlo normalization
- **Change points**: PELT algorithm on entropy/KL/PSI series
- **Outliers**: MAD-based robust z-scores
- **Automated alerts**: Severity-based with evidence JSON

### Probability Models
- **M0 (Baseline)**: Uniform distribution reference
- **M1 (Bayesian)**: Dirichlet-Multinomial with smoothing
- **M2 (Windowed)**: Sliding window with shrinkage
- **Walk-forward evaluation**: Brier score, ECE, lift vs baseline
- **Calibration**: Reliability diagrams and expected calibration error

### Reporting
- **CSV exports**: Detailed metrics and indicators
- **HTML reports**: Printable with mandatory limitations section
- **Interactive UI**: Modern React dashboard with charts

## 📋 Requirements

- **Docker** and **Docker Compose** (recommended)
- OR manually:
  - Python 3.12+
  - PostgreSQL 16+
  - Node.js 20+

## 🚀 Quick Start

### Using Docker Compose (Recommended)

```bash
# Clone or navigate to the repository
cd lotto-analyzer

# Start all services
docker-compose up -d

# Run database migrations
docker-compose exec backend alembic upgrade head

# Generate and load seed data (optional)
docker-compose exec backend python app/db/seed/generate_seed.py --seed 42 --out /tmp/seed.csv
# Then import via UI at http://localhost:5173
```

**Access the application:**
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## 🐧 Déploiement sur Linux (Production)

### Prérequis

```bash
# Installer Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Ajouter l'utilisateur au groupe docker
sudo usermod -aG docker $USER
newgrp docker

# Installer Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Vérifier l'installation
docker --version
docker-compose --version
```

### Déploiement rapide

```bash
# 1. Cloner le repository
git clone https://github.com/votre-repo/lotto-analyzer.git
cd lotto-analyzer

# 2. Créer le fichier .env pour la production
cat > .env << EOF
POSTGRES_USER=lotto
POSTGRES_PASSWORD=$(openssl rand -base64 32)
POSTGRES_DB=lotto_analyzer
CORS_ORIGINS=http://localhost:5173,http://votre-domaine.com
VITE_API_URL=http://localhost:8000
EOF

# 3. Démarrer les services (utiliser docker-compose.prod.yml pour la production)
docker-compose -f docker-compose.prod.yml up -d --build

# 4. Vérifier que les conteneurs sont en cours d'exécution
docker-compose -f docker-compose.prod.yml ps

# 5. Appliquer les migrations de base de données
docker-compose -f docker-compose.prod.yml exec backend alembic upgrade head

# 6. (Optionnel) Générer des données de test
docker-compose -f docker-compose.prod.yml exec backend python app/db/seed/generate_seed.py --seed 42 --out /tmp/seed.csv
```

> **Note**: Utilisez `docker-compose.yml` pour le développement local (avec hot-reload) et `docker-compose.prod.yml` pour la production (build optimisé avec Nginx).

### Configuration avec Nginx (Reverse Proxy)

```bash
# Installer Nginx
sudo apt update && sudo apt install -y nginx

# Créer la configuration
sudo tee /etc/nginx/sites-available/lotto-analyzer << 'EOF'
server {
    listen 80;
    server_name votre-domaine.com;

    # Frontend
    location / {
        proxy_pass http://localhost:5173;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # Backend API
    location /api/ {
        rewrite ^/api/(.*) /$1 break;
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    # SSE (Server-Sent Events) - timeout plus long
    location ~ ^/(games|forensics)/.*/(stream|run/stream) {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Connection '';
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 86400s;
    }
}
EOF

# Activer le site
sudo ln -s /etc/nginx/sites-available/lotto-analyzer /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

### Configuration avec SSL (Let's Encrypt)

```bash
# Installer Certbot
sudo apt install -y certbot python3-certbot-nginx

# Obtenir un certificat SSL
sudo certbot --nginx -d votre-domaine.com

# Renouvellement automatique (déjà configuré par défaut)
sudo systemctl status certbot.timer
```

### Commandes de gestion

```bash
# Voir les logs
docker-compose -f docker-compose.prod.yml logs -f

# Logs d'un service spécifique
docker-compose -f docker-compose.prod.yml logs -f backend

# Redémarrer les services
docker-compose -f docker-compose.prod.yml restart

# Arrêter les services
docker-compose -f docker-compose.prod.yml down

# Arrêter et supprimer les volumes (ATTENTION: perte de données)
docker-compose -f docker-compose.prod.yml down -v

# Mettre à jour l'application
git pull
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d
docker-compose -f docker-compose.prod.yml exec backend alembic upgrade head
```

### Sauvegarde de la base de données

```bash
# Créer une sauvegarde
docker-compose -f docker-compose.prod.yml exec db pg_dump -U lotto lotto_analyzer > backup_$(date +%Y%m%d_%H%M%S).sql

# Restaurer une sauvegarde
cat backup.sql | docker-compose -f docker-compose.prod.yml exec -T db psql -U lotto lotto_analyzer
```

### Systemd Service (Démarrage automatique)

```bash
# Créer le service systemd
sudo tee /etc/systemd/system/lotto-analyzer.service << EOF
[Unit]
Description=Lotto Analyzer Docker Compose
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/chemin/vers/lotto-analyzer
ExecStart=/usr/local/bin/docker-compose -f docker-compose.prod.yml up -d
ExecStop=/usr/local/bin/docker-compose -f docker-compose.prod.yml down
User=$USER

[Install]
WantedBy=multi-user.target
EOF

# Activer le service
sudo systemctl daemon-reload
sudo systemctl enable lotto-analyzer
sudo systemctl start lotto-analyzer
```

### Manual Setup

#### Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure database
cp .env.example .env
# Edit .env with your database URL

# Run migrations
alembic upgrade head

# Start server
uvicorn app.main:app --reload
```

#### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

## 📊 CSV Format

The application expects CSV files with the following format:

```csv
draw_date;n1;n2;n3;n4;n5;bonus
2024-01-03;4;11;19;33;47;8
2024-01-06;2;16;21;27;41;1
2024-01-10;7;13;25;38;42;5
```

**Format specifications:**
- Delimiter: `;` (semicolon)
- Date format: `YYYY-MM-DD` (ISO 8601)
- Numbers: Sorted integers within game rules range
- Bonus: Optional, depends on game configuration
- Encoding: UTF-8

## 🎮 Usage Examples

### 1. Create a Game

```bash
curl -X POST http://localhost:8000/games \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Lotto_5_49",
    "description": "Classic 5/49 lottery with bonus",
    "rules_json": {
      "numbers": {"count": 5, "min": 1, "max": 49, "unique": true, "sorted": true},
      "bonus": {"enabled": true, "min": 1, "max": 10},
      "calendar": {"expected_frequency": "weekly", "days": ["WED", "SAT"]}
    }
  }'
```

### 2. Import Data (Preview)

```bash
curl -X POST "http://localhost:8000/draws/import?game_id=<GAME_ID>&mode=preview" \
  -F "file=@data.csv"
```

### 3. Run Full Analysis

```bash
curl -X POST http://localhost:8000/analyses/run \
  -H "Content-Type: application/json" \
  -d '{
    "game_id": "<GAME_ID>",
    "analysis_name": "full_analysis_v1",
    "params": {}
  }'
```

### 4. Get Probability Forecast

```bash
curl -X POST http://localhost:8000/analyses/run \
  -H "Content-Type: application/json" \
  -d '{
    "game_id": "<GAME_ID>",
    "analysis_name": "forecast_probabilities_v1",
    "params": {"models": ["M0", "M1", "M2"]}
  }'
```

### 5. Export Analysis

```bash
# CSV export
curl http://localhost:8000/analyses/<ANALYSIS_ID>/export.csv -o analysis.csv

# HTML report
curl http://localhost:8000/analyses/<ANALYSIS_ID>/report.html -o report.html
```

## 🧪 Testing

```bash
cd backend

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test
pytest tests/test_randomness.py -v
```

**Test coverage includes:**
- Import validation and CSV parsing
- χ² tests and FDR correction
- Change point detection on synthetic data
- M0/M1/M2 model scoring and calibration
- Walk-forward evaluation

## 📁 Project Structure

```
lotto-analyzer/
├── backend/
│   ├── app/
│   │   ├── api/              # FastAPI routes
│   │   ├── services/         # Business logic
│   │   ├── analysis/         # Analysis engine
│   │   │   ├── metrics.py
│   │   │   ├── randomness.py
│   │   │   ├── anomalies.py
│   │   │   ├── prob_models/  # M0, M1, M2
│   │   │   ├── evaluation/   # Walk-forward, Brier, ECE
│   │   │   └── reporting/    # CSV, HTML exports
│   │   ├── db/               # Models, migrations
│   │   │   └── seed/         # Synthetic dataset
│   │   └── schemas/          # Pydantic models
│   ├── tests/                # Pytest tests
│   ├── alembic/              # Database migrations
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pages/            # Dashboard, Import, Analysis, etc.
│   │   ├── api/              # API client
│   │   └── App.tsx
│   └── package.json
├── docker-compose.yml
└── README.md
```

## 🔬 Scientific Methodology

### Hypothesis Testing
- **H0**: Lottery follows uniform i.i.d. process without replacement
- **Multiple testing**: FDR correction (Benjamini-Hochberg)
- **Meta-tests**: P-value distribution analysis (KS test)

### Probability Evaluation
- **Walk-forward**: No temporal leakage, train on past only
- **Brier score**: Multi-label probability accuracy
- **ECE**: Expected calibration error
- **Baseline comparison**: Always compare to uniform M0

### Anomaly Detection
- **PSI thresholds**: 0.10 (low), 0.25 (medium), 0.50 (high)
- **KL normalization**: Monte Carlo simulation (n=1000, seed=42)
- **MAD z-scores**: Robust outlier detection (threshold=3)
- **Change points**: PELT with penalty tuning

## ⚠️ Limitations & Interpretation (MANDATORY)

**This application is for scientific analysis only. Key limitations:**

1. **No prediction guarantees**: A healthy lottery is fundamentally random
2. **Baseline comparison**: All models must be compared to uniform distribution (M0)
3. **Multiple testing**: Apparent patterns may be false positives
4. **Sample size**: Small datasets reduce statistical power
5. **Temporal changes**: Rule changes, data quality issues affect validity
6. **No gaming advice**: This tool does NOT provide "winning numbers"

**Proper interpretation requires:**
- Understanding of statistical hypothesis testing
- Awareness of multiple comparison problems
- Recognition that past draws don't influence future draws
- Critical evaluation of model lift vs baseline

## 🔧 Configuration

### Game Rules Example

```json
{
  "numbers": {
    "count": 5,
    "min": 1,
    "max": 49,
    "unique": true,
    "sorted": true
  },
  "bonus": {
    "enabled": true,
    "min": 1,
    "max": 10
  },
  "calendar": {
    "expected_frequency": "weekly",
    "days": ["WED", "SAT"]
  }
}
```

### Analysis Parameters

```json
{
  "window_size": 200,
  "lambda_shrink": 0.3,
  "monte_carlo_sims": 1000,
  "fdr_method": "fdr_bh",
  "change_point_penalty": 3
}
```

## 📈 Seed Dataset

Generate reproducible synthetic dataset:

```bash
cd backend
python app/db/seed/generate_seed.py --seed 42 --out seed_data.csv
```

**Synthetic scenarios included:**
- **S0**: Normal uniform draws (2 years)
- **S1**: Mild drift (3 months, weight=1.2 on [7,13,42])
- **S2**: Rupture (1 month, weight=1.8 on [7,13,42])
- **S3**: Data quality issues (missing, invalid, duplicates)
- **S4**: Outliers (extreme sums)

## 🛠️ Development

### Code Quality

```bash
# Format code
black app/

# Lint
ruff check app/

# Type checking (optional)
mypy app/
```

### Database Migrations

```bash
# Create new migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

## 📚 API Documentation

Interactive API documentation available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🤝 Contributing

This is a research/thesis project. Key principles:
1. **Scientific rigor**: All methods must be statistically sound
2. **Reproducibility**: Fixed seeds, versioned code/data
3. **Transparency**: Clear limitations and assumptions
4. **Testing**: All statistical methods must have tests

## 📄 License

This project is for educational and research purposes.

## 🔗 References

- Benjamini-Hochberg FDR correction
- Brier score for probability evaluation
- PELT algorithm for change point detection
- Population Stability Index (PSI)
- Kullback-Leibler divergence
- Dirichlet-Multinomial distribution

## 📞 Support

For issues or questions related to:
- **Statistical methodology**: See `cdc/01_methodology.md`
- **API contracts**: See `cdc/03_api_spec.md`
- **Alert policies**: See `cdc/04_alerting_policy.md`
- **Model catalog**: See `cdc/06_model_catalog.md`

---

**Version**: v1.0.0  
**Last Updated**: 2025-01-23  
**Status**: Production-ready for research use
