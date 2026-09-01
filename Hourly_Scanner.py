name: Nifty 500 Hourly Scanner

on:
  workflow_dispatch:

  schedule:
    - cron: "45 4 * * 1-5"
    - cron: "45 5 * * 1-5"
    - cron: "45 6 * * 1-5"
    - cron: "45 7 * * 1-5"
    - cron: "45 8 * * 1-5"
    - cron: "45 9 * * 1-5"

jobs:

  scan:

    runs-on: ubuntu-latest

    steps:

      # ------------------------------------------------------
      # CHECKOUT REPOSITORY
      # ------------------------------------------------------

      - name: Checkout repository
        uses: actions/checkout@v4


      # ------------------------------------------------------
      # SETUP PYTHON
      # ------------------------------------------------------

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"


      # ------------------------------------------------------
      # INSTALL DEPENDENCIES
      # ------------------------------------------------------

      - name: Install dependencies
        run: |
          pip install -r requirements.txt


      # ------------------------------------------------------
      # RUN NIFTY 500 SCANNER
      # ------------------------------------------------------

      - name: Run hourly scanner
        run: |
          python Hourly_Scanner.py


      # ------------------------------------------------------
      # CHECK SCANNER RESULT
      # ------------------------------------------------------

      - name: Check scanner result
        run: |

          if [ -f "hourly_signals.csv" ]; then

            echo "hourly_signals.csv FOUND"

            ls -lh hourly_signals.csv

            echo "----------------------------------------"
            echo "SCANNER RESULTS"
            echo "----------------------------------------"

            cat hourly_signals.csv

          else

            echo "hourly_signals.csv NOT FOUND"

          fi


      # ------------------------------------------------------
      # SEND RESULTS BY GMAIL
      # ------------------------------------------------------

      - name: Send results by Gmail

        env:

          GMAIL_USERNAME: ${{ secrets.GMAIL_USERNAME }}

          GMAIL_APP_PASSWORD: ${{ secrets.GMAIL_APP_PASSWORD }}

          GMAIL_TO: ${{ secrets.GMAIL_TO }}

        run: |

          python - <<'PY'

          import os
          import smtplib

          from pathlib import Path

          from email.message import EmailMessage


          # --------------------------------------------------
          # GMAIL SETTINGS
          # --------------------------------------------------

          username = os.environ["GMAIL_USERNAME"]

          app_password = os.environ["GMAIL_APP_PASSWORD"]

          recipient = os.environ["GMAIL_TO"]


          # --------------------------------------------------
          # SCANNER OUTPUT FILE
          # --------------------------------------------------

          file_path = Path("hourly_signals.csv")


          # --------------------------------------------------
          # CREATE EMAIL
          # --------------------------------------------------

          email = EmailMessage()

          email["From"] = username

          email["To"] = recipient

          email["Subject"] = "NIFTY 500 Hourly Scanner Results"


          # --------------------------------------------------
          # IF MATCHING STOCKS WERE FOUND
          # --------------------------------------------------

          if file_path.exists():

              email.set_content(
                  "
