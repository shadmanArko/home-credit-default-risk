# AWS deployment guide (`HC-M4-15`–`18`)

How the serving API (Chunk 3/3B) gets deployed to AWS Lambda behind a
free, public HTTPS endpoint, at ~$0/month for demo-level traffic.

## Why this design

- **AWS Lambda over SageMaker / ECS Fargate / EC2**: Lambda's always-free
  tier (1M requests + 400,000 GB-seconds/month) fits intermittent demo
  traffic at $0. The others bill continuously whether or not anyone hits
  the endpoint.
- **Container image, not a zip package**: the app already has a working
  Docker image (Chunk 3) with real dependencies (LightGBM, DuckDB,
  MLflow) that don't fit Lambda's zip-package size limits comfortably.
  Container-image Lambda has no such limit (up to 10 GB).
- **AWS Lambda Web Adapter, not Mangum**: the adapter is a Lambda
  extension that proxies invocations to a normal HTTP server. The
  existing FastAPI app (`adapters/api/main.py`) runs completely
  unchanged — same `uvicorn` command as local Docker Compose. See
  `Dockerfile.lambda`'s header comment.
- **AWS SAM, not Terraform/CDK**: purpose-built for exactly this shape
  (container image → Lambda → API Gateway) with the least new tooling
  to learn for this scope.
- **The deployed model is the real, locally-trained one, baked into the
  image at build time — never retrained or served from CI's synthetic
  fixture.** Lambda containers have no bind mounts, so
  `docker compose run --rm lambda-fixture` produces `lambda_build/`
  (gitignored) on your machine first, and `Dockerfile.lambda` `COPY`s it
  into the image. Re-run that fixture step and rebuild/redeploy whenever
  the model changes.

## What only you can do (account, billing, credentials)

I will not create the account, view/enter payment details, or handle
your AWS credentials in any way — you run every step in this section
yourself, in your own terminal / AWS console.

1. **Sign up for AWS** (if you haven't already) at
   https://aws.amazon.com — free tier, requires a card on file but this
   deployment is designed to stay within the always-free Lambda/API
   Gateway/CloudWatch tiers.
2. **Set a billing alarm/budget first, before creating anything else.**
   AWS Console → Billing → Budgets → create a budget (e.g. $5/month)
   with an email alert. This is the actual senior-engineer move here —
   catch a misconfiguration before it costs money, not after.
3. **Create an IAM user for CLI access** (never use your root account
   for this):
   - IAM → Users → Create user → programmatic access only (access
     key + secret key, no console password needed).
   - Attach a scoped policy rather than `AdministratorAccess`. For this
     project's resources specifically, an inline policy covering:
     `cloudformation:*`, `lambda:*`, `apigateway:*`, `ecr:*`, `iam:*Role*`
     `iam:*Policy*` (SAM needs to create the execution role),
     `logs:*`, `s3:*` (SAM's deployment artifact bucket), scoped to your
     account/region is enough. If you'd rather move faster for a
     personal demo project and tighten later, that's your call — just
     know the tradeoff.
   - Download the access key + secret key (shown once).
4. **Configure the AWS CLI with those keys** — run this yourself, I will
   never see or type these values:
   ```bash
   aws configure
   ```
   It asks for: Access Key ID, Secret Access Key, default region (pick
   one close to you, e.g. `us-east-1`), default output format (`json` is
   fine).
5. Confirm it worked:
   ```bash
   aws sts get-caller-identity
   ```
6. **Create a small S3 bucket for the real model artifacts** (`HC-M4-17`):
   `lambda_build/` (the real, locally-trained model + feature store) is
   gitignored — same convention as `mlflow.db`/`mlruns/` elsewhere in
   this project — so GitHub Actions has no way to check it out. Instead,
   it's synced to a private S3 bucket once as part of your manual
   deploy, and CD downloads it from there on every automated redeploy.
   ```bash
   aws s3 mb s3://<a-globally-unique-bucket-name>
   ```
   Keep the bucket private (the default) — it never needs public access.
7. **Add repo secrets** for CD (GitHub repo → Settings → Secrets and
   variables → Actions → New repository secret):
   `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`,
   `LAMBDA_ARTIFACTS_BUCKET` (the bucket name from step 6). Until these
   are added, the `deploy` job in `ci.yml` simply fails harmlessly on
   every merge to `main` — nothing destructive happens either way.

## What I've already done, verified without touching AWS

- `Dockerfile.lambda` — the Lambda-deployable image, verified with a
  real `docker build`.
- `docker-compose.yml`'s `lambda-fixture` service — produces
  `lambda_build/` from the real dataset. **Already run once**; re-run
  with `docker compose run --rm lambda-fixture` whenever the model needs
  refreshing.
- `template.yaml` (SAM) — validated with `sam validate --lint`.
- `sam build` — builds the real container image locally (`.aws-sam/build`).
- **`sam local start-api`** — ran the actual built image locally,
  emulating Lambda + API Gateway, and hit `/health`, `/score` (a real
  known applicant, `SK_ID_CURR=100002`, scored `0.83`, high risk), and
  `/apply` (a new applicant, scored `0.13`, low risk) — all real 200s
  with real probabilities from the real model. This is the strongest
  pre-AWS confidence check available: the exact image that will ship.
- Along the way, found and fixed two real container-sizing bugs the
  local emulation caught before they'd have cost money or failed
  silently in production:
  - **OOM at the default 1024 MB** — `LocalFeatureStore` loads all
    356,255 applicants into an in-memory dict at startup; fixed by
    raising `MemorySize` to 3008 MB in `template.yaml`.
  - **Cold start exceeding a 30s timeout** — loading the feature store
    + opening the MLflow SQLite registry takes close to 30s on a cold
    container; fixed by raising `Timeout` to 60s.

## The actual deploy (once you've done the account/IAM/`aws configure` steps)

```bash
# Only needed once, or whenever the real model/feature store changes:
docker compose run --rm lambda-fixture

# Push the fresh artifacts to S3 so future automated CD redeploys
# (HC-M4-17) have something to build from -- lambda_build/ itself is
# gitignored and never reaches GitHub Actions any other way.
aws s3 sync lambda_build/ s3://<your-bucket-name>/lambda_build/ --delete

sam build
sam deploy --guided
```

`sam deploy --guided` will ask a handful of questions the first time
(stack name, region, "allow SAM to create IAM roles" → yes) and then
save them to `samconfig.toml` so future deploys are just `sam deploy`.

It will create, via CloudFormation: an ECR repository, push the built
image to it, create the Lambda function (`PackageType: Image`, the
default `AWSLambdaBasicExecutionRole` — CloudWatch Logs only, since this
function calls no other AWS service), and an HTTP API Gateway in front
of it with a catch-all proxy route.

At the end it prints the `ApiUrl` output — a real
`https://<id>.execute-api.<region>.amazonaws.com/` URL. Test it exactly
like the local smoke test:

```bash
curl https://<your-api-url>/health
curl -X POST https://<your-api-url>/score -H "Content-Type: application/json" -d '{"sk_id_curr": 100002}'
```

Open `https://<your-api-url>/` in a browser for the demo frontend
(`HC-M4-23`).

**Verify CloudWatch Logs**: AWS Console → CloudWatch → Log groups →
`/aws/lambda/<function-name>` — should show the same structured
`INFO:home_credit_api:...` lines seen locally.

**Verify $0 spend**: AWS Console → Billing → Cost Explorer, a day or two
after deploying. Demo-level traffic should round to $0 given the always-
free tiers.

## Tearing it down (stop all ongoing cost)

```bash
sam delete
```

Removes the Lambda function, API Gateway, ECR repository, and the
CloudFormation stack. Nothing continues to bill after this.

## Redeploying after a code change (manual, until `HC-M4-17`'s CD wiring)

```bash
sam build
sam deploy
```

This rebuilds and redeploys the *code* — it does **not** retrain the
model. If the model itself changed, re-run
`docker compose run --rm lambda-fixture` and the `aws s3 sync` step
above first, so the new artifacts are baked into the image before
`sam build` (and so the next automated CD run in `ci.yml` picks up the
new model too, not just the old one from S3).

Once the repo secrets from step 7 above are set, merging to `main`
triggers `HC-M4-17`'s `deploy` job automatically: it downloads whatever
is currently in the S3 bucket, rebuilds `Dockerfile.lambda` with your
latest code, and redeploys — a genuine code-only CD path that never
retrains or touches the real dataset.
