# План работ BubbleTeaIT

Состояние на 27 октября 2026. Полная история — в `attic/TODO-history.md`
(локально, вне Git). Здесь только текущий baseline и ближайшие задачи.

## P0. Чистый baseline и воспроизводимость

- [x] Вынести исторические документы, benchmark-скрипты и старые тесты
  в локальный `attic/`; в Git оставить только продуктовый костяк.
- [x] Сократить `tests/` до набора, который проверяет текущий продукт и
  не зависит от локальных `evals/`.
- [x] Закоммитить baseline (`62de9d0`) и проверить чистый clone:
  268 tests OK, 12 skipped.
- [x] Добавить CI (Windows, Python 3.12): `.github/workflows/ci.yml`
  устанавливает `.[mcp]`, запускает `doctor` и `unittest`; первый прогон
  будет после push.
- [x] Закрепить в README только существующие документы; мёртвых ссылок нет.
- [x] Проверить `git submodule update --init --depth 1` из удалённого origin:
  `vendor/cocoindex-code` скачан в чистом clone на закреплённую ревизию;
  остальные URL доступны, ревизии заданы gitlink.

## P1. Баги сервиса

- [x] `sync(wait=False)` возвращает `updating`, пока новая ревизия не
  опубликована; регрессия в `test_service`.
- [x] `stop` возвращает `stopping`, если координатор ещё не завершился;
  `ensure` ждёт `service.lock`, а не падает с `busy`.
- [x] Windows-spawn больше не держит `service.lock` во время WMI/cscript.
- [x] `heartbeat` возвращает `found`; клиент перерегистрирует проект после
  потери lease.
- [x] Битый `state.json` карантинится, `status`/`ensure` видят «stopped».
- [x] `fingerprint` дополнен дешёвой проверкой readiness индексов; удалённый
  индекс пересобирается, а не остаётся `ready`.
- [x] `register`/`project_info` выполняют тяжёлую валидацию вне `cv`.
- [x] `freshness=wait` подтверждает неизменный проект без ожидания build
  другого проекта.
- [x] Смешанный batch ID+path без `composition_id` отклоняется.

## P1. Fidelity и границы

- [x] `SearchService._decorate` требует точный срез
  `lines[start-1:end]`, а не подстроку; content строится из snapshot.
- [x] `configuration.source_roots` сверяется с `provenance.source_root`;
  чужой root отклоняется.
- [x] Alias и logical `source_id` не пересекаются; коллизия — ошибка.
- [x] `_bytes` проверяет длину файла; `check-ignore` больше не теряет
  tracked-но-ignored файлы.
- [ ] Text-канал: не молча пропускать non-UTF8; сообщать skipped/unsearchable.
- [ ] Text-канал: склеивать пересекающиеся окна, иначе один файл вытесняет
  остальные из top-k.
- [ ] Вынести общий verifier результата backend'а для `search`/`cocoindex`/`qwen`.

## P1. Производительность

- [ ] Не верифицировать все байты корпуса на каждый поиск/чтение: кэш
  verified-манифеста по immutable `unit_id` + manifest hash; побайтовая
  проверка только читаемого unit'а.
- [ ] `index_status` не должен вызывать `load_unit(verify_files=True)`.
- [ ] `_decorate` не должен перечитывать файл на каждую строку; кэш
  `path -> (lines, sha)` на время запроса.
- [ ] Poll: stat-fast-path `(path,size,mtime_ns)` до sha256; дефолт
  `poll_interval_seconds` 10–30 с вместо 1 с.
- [ ] `git check-ignore` батчами, а не subprocess на каждый каталог.
- [ ] `preview` не должен читать каждый файл дважды.
- [ ] Упаковка direct: не сериализовать весь ответ на каждый вариант окна.
- [ ] Замерить p95 search/read на среднем и крупном корпусе; зафиксировать бюджет.

## P2. Рефакторинг и упрощение

- [x] Вынести замороженный `research_*` runtime из пакета; в пакете оставить
  только `serena_workspace`/`pinned`, которые нужны `symbols`.
- [x] Переименовать `research_serena.py`/`research_tools.py` в
  `serena_workspace.py`/`pinned.py` и обновить импорты.
- [ ] Убрать из `config.py` разбор `[research]` после миграции локальных
  конфигов (сейчас сохранён для совместимости).
- [ ] Разделить `build_server` (284 строки) на регистрацию доменных групп.
- [ ] Свести приоритет конфигов к одной документированной схеме; убрать
  расхождение между docstring, `setup.md` и `service.md`.
- [ ] Один verifier результата backend'а вместо дублей в `search`,
  `cocoindex`, `qwen`.
- [x] Registry: writer lock для `register`/`compose`.
- [ ] Убрать mutable side-channel `_last_selection_rules`; GC staging/orphan
  units и retention старых compositions.
- [ ] Сохранять причину исключения файла (`skipped`) в manifest, а не
  только в `preview`.
- [x] `check-ignore` больше не теряет tracked-но-ignored файлы; добавлен
  регрессионный тест.

## P2. Документация

- [x] Убрать из Git исторические концепции/слайсы и проектные документы;
  оставить только общий роадмап и эксплуатационные доки.
- [x] Сократить `setup.md` до установки/запуска.
- [x] Синхронизировать список MCP-инструментов (13) с фактом.
- [ ] Решить судьбу `evals/retrieval-contract-v1` и `symbols-contract-v1`:
  либо публичные case-файлы в Git, либо явные skip-guards в тестах.

## P3. Доказательство пользы

- [ ] Независимый holdout: 20–30 интентов (код/документы/межпродуктовые),
  зафиксированные пороги до прогона.
- [ ] Сравнение с обычным `rg`/чтением при равных бюджетах; считать
  task success, hit@5/10, токены, wall time.
- [ ] Не менять инструмент во время контрольного прогона.
- [ ] Повторить на реальном корпусе после исправлений retrieval.

## Отложено: исследователь

- [ ] Вернуть `research_*` слой после стабилизации прямого BTI. Код, тесты,
  worker и bridge лежат в `attic/src`, `attic/tests`, `attic/tools`;
  `research_serena`/`research_tools` уже стали `serena_workspace`/`pinned`.
- [ ] Условия возврата: независимый benchmark прямого поиска, стабильный
  service/MCP, отдельный профиль зависимостей (`pydantic-ai`/Codex CLI),
  решение по вторичным MCP и бюджетам.
- [ ] При возврате восстановить `research_start/status/cancel`, CLI `research`,
  конфиг `[research]` и проверить, что вторичные MCP действительно доступны.
