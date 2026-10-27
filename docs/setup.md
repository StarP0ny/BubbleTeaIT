# Установка и запуск

Нужны Python 3.12+ и Git. Модельные веса и индексы в пакет не входят.

## Окружение

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[mcp]'
```

`[mcp]` нужен для stdio-фасада. Для старого адаптера CocoIndex/E5 создайте
отдельное окружение, чтобы не смешивать ML-зависимости с основным venv:

```powershell
python -m venv .venv-indexing
.\.venv-indexing\Scripts\python.exe -m pip install -e '.[indexing]'
```

Путь к этому Python указывается в `backends.cocoindex.python_executable`.

Qwen3-Embedding работает в отдельном worker-окружении с `torch`,
`transformers` и `sentence-transformers`; путь к его Python задаётся в
конфиге backend, а не в основном venv.

## Данные и конфиг

Runtime-данные по умолчанию:

- Windows: `%LOCALAPPDATA%/BubbleTeaIT`;
- XDG: `$XDG_DATA_HOME/bubbleteait`.

Переопределение: `BTI_DATA_DIR`, `BTI_MODELS_DIR`, `--config`,
`--local-config`. Приоритет настроек и машинный конфиг сервиса описаны в
[service.md](service.md); проектные настройки — в
[project-config.md](project-config.md).

```powershell
.\.venv\Scripts\python.exe -m bubbleteait init
.\.venv\Scripts\python.exe -m bubbleteait doctor
.\.venv\Scripts\python.exe -m bubbleteait doctor --json
```

`doctor` без `--probe-endpoints` не ходит в сеть. Проверка файлов модели не
загружает её.

## Проект

В корне проекта создайте `bti.toml` (пример — [project-config.md](project-config.md))
и выполните:

```powershell
bti plan
bti sync
bti project-info
bti search "Как отключается нагрев при ошибке датчика?" --channel vector
```

`plan` не регистрирует источники и не загружает модели. `sync` создаёт снимки,
индексирует отсутствующие units и публикует состав. Повторный `sync` без
изменений переиспользует готовые units.

## Сервис

```powershell
bti service start
bti service status
bti service config show
bti service stop
```

Watch и workspace:

```powershell
bti watch -p C:/work/controller
bti watch --list
bti workspace list
bti workspace status example
```

Сервис перечитывает `config-service.toml` автоматически. Остановка не удаляет
снимки и регистрации. Подробности — в [service.md](service.md).

## MCP

```powershell
.\.venv\Scripts\python.exe -X utf8 -m bubbleteait --project-dir C:/work/controller mcp
```

Пример подключения клиента:

```json
{
  "mcpServers": {
    "bti": {
      "command": "C:/path/to/.venv/Scripts/python.exe",
      "args": ["-X", "utf8", "-m", "bubbleteait", "--project-dir", "C:/work/controller", "mcp"]
    }
  }
}
```

Первый `context_search` сам подключает сервис и при необходимости
синхронизирует проект. `read_sources` читает известный путь или выданный ID.

## Qwen worker

```powershell
bti worker status
bti worker load
bti worker unload
bti worker stop
```

Первый vector-запрос загружает модель; последующие используют ту же копию.
Модель должна лежать в `models_dir`, иметь `bti-model-manifest.json` со
`status="complete"` и закреплённую ревизию. Профиль `qwen-dense` фиксирован на
Qwen3-Embedding-0.6B; замена строки `model_path` не делает другую модель
поддержанной.

## Проверки

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -X utf8 tools/probe_mcp.py --composition <id> --query "<query>" --channel text --output .local/mcp-check.json
```

Baseline-набор не требует облачной модели и сети. Skips относятся к
необязательным окружениям (numpy/CUDA/Windows/внешние бинарники) и не
заменяют проверку качества поиска.
