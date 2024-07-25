# Overview
While database access is at the moment not safe for concurrent requests, the web server implements the relevant guardrails. Be sure that only one web server is running at a time.

```bash
# Capture the hostname of the submit host
# Export it as an environment variable and submit the job
SUBMIT_HOST=$(hostname)
sbatch --export=SUBMIT_HOST=$SUBMIT_HOST scripts/ai_preference.sh
```
