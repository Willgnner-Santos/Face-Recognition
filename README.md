# Reconhecimento Facial

## Pré-requisitos
- Docker e Docker Compose instalados.
- Python 3.10+ para rodar o notebook.

## Subir os serviços
1. Build e subida dos containers:

```bash
docker compose up --build
```

2. A API ficará disponível em `http://localhost:5000`.

## Rodar o notebook para testar e cadastrar
1. Instale as dependências locais do notebook:

```bash
pip install -r requirements_teste.txt
```

2. Inicie o Jupyter:

```bash
jupyter notebook
```

3. Abra o notebook `teste_fotos_quantize.ipynb`.

4. Execute as células na ordem:
- A primeira célula define `API_BASE_URL = "http://localhost:5000"` e `DATASET_PATH = "images/FOTOS JPG"`.
- A célula `build_dataset_and_register(DATASET_PATH, API_BASE_URL)` cadastra a primeira imagem de cada pessoa.
- As células de `process_predictions_for_subset(...)` executam as predições e permitem avaliar métricas.

5. Para testar com base externa, use o bloco:
- `test_path = "images/DATASET WEBCAM"` e execute a célula que monta o `df_test_extra`.

## Observações
- Mantenha o `docker compose` rodando enquanto executa o notebook.
- Se quiser usar outro dataset, ajuste `DATASET_PATH` no notebook.
