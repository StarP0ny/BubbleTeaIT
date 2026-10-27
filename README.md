# BubbleTeaIT

Локальный слой контекста для AI-агентов: индексирует проекты и документы,
хранит неизменяемые снимки точных версий и отдаёт агенту проверяемые
фрагменты с путями, строками и SHA — через CLI `bti` и MCP.

Проект не зависит от VS Code и конкретного харнеса. Индекс версии модуля
переиспользуется несколькими составами; внутренние базы движков не сливаются.

## Статус

Рабочий baseline:

- регистрация источников, immutable-снимки и составы точных версий;
- семантический поиск Qwen3-Embedding 0.6B и старый адаптер CocoIndex/E5;
- прямой поиск и чтение с цитируемыми evidence-ID;
- навигация по C/C++ через standalone Serena;
- локальный сервис с watch, автосинхронизацией и workspace-поиском;
- stdio MCP.

Направление исследователя вынесено из baseline в локальный `attic/`:
его код, эксперименты и тесты не входят в Git. Качество поиска на независимой
выборке пока не доказано: см. раздел «Ограничения».

## Быстрый старт

Нужен Python 3.12+ и Git.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[mcp]'
.\.venv\Scripts\python.exe -m bubbleteait init
.\.venv\Scripts\python.exe -m bubbleteait doctor
```

В проекте создаётся `bti.toml` (см. [конфиги проектов](docs/project-config.md)),
затем:

```powershell
bti plan
bti sync
bti project-info
```

Для постоянного наблюдения и MCP:

```powershell
bti watch -p C:/work/controller
bti mcp --project-dir C:/work/controller
```

Установка `bti` в PATH и настройка моделей — в [инструкции запуска](docs/setup.md).

## MCP

Stdio-фасад предоставляет 13 инструментов:

- прямой контекст: `context_search`, `read_sources`, `resolve_sources`;
- навигация: `list_sources`, `list_files`, `code_symbols`;
- точный доступ: `search`, `read_source`, `index_status`, `project_info`;
- сервис: `watch_status`, `sync_status`, `sync_project`.

Рабочий маршрут агента: `context_search` → при необходимости `read_sources`.
Каталог и `resolve_sources` нужны только для навигации и метаданных.

Сервис, lease, workspace и `freshness=wait` описаны в
[документе сервиса](docs/service.md).

## Данные и границы

- Runtime-данные по умолчанию лежат вне checkout:
  `%LOCALAPPDATA%/BubbleTeaIT` (Windows) или `$XDG_DATA_HOME/bubbleteait`.
- `.local/` — локальные снимки, прогоны и receipts; в Git не входит.
- `docs/research/`, `docs/evals/`, `evals/`, `profiles/`, `infra/` —
  локальные исследовательские материалы; в baseline не входят.
- `attic/` — вынесенные из baseline исторические документы, тесты и
  benchmark-скрипты. Файлы остаются на диске, но не коммитятся.
- Веса моделей, индексы и секреты в Git не хранятся.

## Компоненты

Исходники сторонних компонентов закреплены Git-сабмодулями:

```powershell
git submodule update --init --depth 1
```

Список и ревизии — в [components.md](docs/components.md). Наличие сабмодуля
не означает, что сервис установлен или запущен; установка выполняется по
инструкциям upstream.

## Проверки

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Baseline-набор не требует облачной модели, GPU или сети. Часть тестов
использует фикстуры реальных снимков; отдельные проверки могут пропускаться
при отсутствии необязательных зависимостей — skips нужно учитывать отдельно.

Проверка реального stdio-клиента:

```powershell
.\.venv\Scripts\python.exe -X utf8 tools/probe_mcp.py --composition <composition-id> --query "heating control" --channel text --output .local/mcp-check.json
```

## Ограничения

- Качество поиска не доказано на независимом holdout; текущие наборы —
  development-наборы. Подробности и отрицательные результаты сохранены
  локально, а не выдаются за приёмку.
- Публичный `search`/CLI проверяют диапазон и содержимое результата слабее,
  чем direct-путь; это известный дефект, см. [TODO](TODO.md).
- Сервис использует периодическую сверку корпуса, а не файловые уведомления
  ОС; стоимость сверки больших SDK не измерена.
- Исследователь не входит в baseline; его материалы остаются локальными.

## Документы

- [Иерархические конфиги](docs/project-config.md).
- [Сервис и watch](docs/service.md).
- [Установка и модели](docs/setup.md).
- [Компоненты и ревизии](docs/components.md).
- [План работ](TODO.md).
- [Правила работы с репозиторием](AGENTS.md).
