set -e

echo "LiveYDream Local Setup with Supabase"
echo "========================================"

if [ ! -f .env ]; then
    echo ".env file not found. Creating from .env.example..."
    cp .env.example .env
    echo "Please edit .env and add your Supabase password!"
    echo "   DB_PASSWORD=your_password_here"
    exit 1
fi

echo ""
echo " Testing Supabase connection..."
if command -v psql &> /dev/null; then
    DB_HOST=$(grep DB_HOST .env | cut -d'=' -f2)
    DB_USER=$(grep DB_USER .env | cut -d'=' -f2)
    DB_NAME=$(grep DB_NAME .env | cut -d'=' -f2)
    echo "   Host: $DB_HOST"
    echo "   User: $DB_USER"
    echo "   DB:   $DB_NAME"
else
    echo "   psql not installed. Skip connection test."
fi
echo ""
echo " Starting Docker services (Redis, MinIO, MLflow, Bull Board)..."
docker compose -f docker-compose.local.yml up -d

echo ""
echo " Waiting for services to be healthy..."
sleep 5

echo ""
echo " Services status:"
docker compose -f docker-compose.local.yml ps

echo ""
if [ -f package.json ]; then
    echo " Installing Node.js dependencies..."
    npm install
fi

echo ""
if [ -f ml-models/requirements.txt ]; then
    echo " Installing Python dependencies..."
    pip install -r ml-models/requirements.txt 2>/dev/null || pip3 install -r ml-models/requirements.txt
fi


echo ""
echo "========================================"
echo " Setup complete!"
echo ""
echo " Next steps:"
echo "   1. Add your Supabase password to .env"
echo "      DB_PASSWORD=your_supabase_password"
echo ""
echo "   2. Run the SQL schema on Supabase:"
echo "      Go to: https://supabase.com/dashboard/project/ljbkjafykyiqmjtwlwpn/sql"
echo "      Paste the contents of: 001_supabase_schema.sql"
echo ""
echo "   3. Start the backend:"
echo "      npm run start:dev"
echo ""
echo " Services:"
echo "   • Backend:     http://localhost:3000"
echo "   • Bull Board:  http://localhost:3001"
echo "   • MLflow:      http://localhost:5000"
echo "   • MinIO:       http://localhost:9001 (minioadmin/minioadmin)"
echo ""
