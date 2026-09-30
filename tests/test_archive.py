"""连续前缀归档测试：清除明细载荷、墓碑、水位约束与幂等。

覆盖：归档清空观测与阈值只留墓碑（batch_id/序号/前后摘要/条数/时间/首次响应/
重放统计）、目标不得超过同事务核定的 safe_sequence（越界 409 附水位与阻塞
消费者）、重复或较小目标幂等、链头摘要与后续上传不变、已归档批次的同载荷
重放（累计重放次数）与载荷变化 409、正确摘要确认幂等且不倒退水位、归档后
登记的新消费者从 archived_through 起步（首个缺口指向下一批）、并发归档/
确认/登记不越过未确认批次且水位不回退，以及旧库迁移补 archived_through 列。
"""
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

THRESHOLDS = {"tolerance": 0.5, "max_drift": 5e-4, "max_missed": 3, "max_false": 3}
DEVICE = "dev-01"


@pytest.fixture()
def client(tmp_path):
    app = create_app(db_path=str(tmp_path / "test.db"))
    with TestClient(app) as c:
        yield c


def obs(n, base=1_700_000_000.0, step=0.5):
    return [
        {"local_id": f"o{i:03d}", "captured_at": base + i * step}
        for i in range(n)
    ]


def batch(batch_id, sequence, previous_digest="", observations=None, **kw):
    body = {
        "batch_id": batch_id,
        "sequence": sequence,
        "previous_digest": previous_digest,
        "observations": observations if observations is not None else obs(3),
        **THRESHOLDS,
    }
    body.update(kw)
    return body


def upload_chain(client, device, n):
    """上传 n 个链式批次，返回 (各批次摘要列表, 各批次原始响应列表)。"""
    digests, responses = [], []
    prev = ""
    for seq in range(1, n + 1):
        r = client.post(
            f"/field-devices/{device}/batches",
            json=batch(f"b-{seq}", seq, previous_digest=prev),
        )
        assert r.status_code == 201
        digests.append(r.json()["digest"])
        responses.append(r.json())
        prev = digests[-1]
    return digests, responses


def register(client, device, consumer_id):
    return client.post(
        f"/field-devices/{device}/consumers", json={"consumer_id": consumer_id}
    )


def ack(client, device, consumer_id, sequence, digest):
    return client.post(
        f"/field-devices/{device}/acknowledgments",
        json={"consumer_id": consumer_id, "sequence": sequence, "batch_digest": digest},
    )


def ack_all(client, device, consumer_id, digests, upto=None):
    for seq in range(1, (upto or len(digests)) + 1):
        r = ack(client, device, consumer_id, seq, digests[seq - 1])
        assert r.status_code in (200, 201)


def archive(client, device, through_sequence):
    return client.post(
        f"/field-devices/{device}/archive", json={"through_sequence": through_sequence}
    )


def status(client, device):
    return client.get(f"/field-devices/{device}/archive-status").json()


# ---------------- 归档：墓碑化与响应字段 ----------------


def test_archive_tombstones_clear_payload_keep_envelope(client):
    digests, responses = upload_chain(client, DEVICE, 4)
    register(client, DEVICE, "archiver")
    register(client, DEVICE, "ml-pipeline")
    ack_all(client, DEVICE, "archiver", digests)
    ack_all(client, DEVICE, "ml-pipeline", digests)
    assert status(client, DEVICE)["safe_sequence"] == 4

    r = archive(client, DEVICE, 2)
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "archived"
    assert body["device_id"] == DEVICE
    assert body["through_sequence"] == 2
    assert body["archived_through"] == 2
    assert body["archived_count"] == 2
    assert body["safe_sequence"] == 4

    # 列表：前缀内批次为墓碑 archived，其余仍 accepted；响应带 archived_through
    lst = client.get(f"/field-devices/{DEVICE}/batches").json()
    assert lst["archived_through"] == 2
    by_seq = {b["sequence"]: b for b in lst["batches"]}
    assert by_seq[1]["status"] == "archived"
    assert by_seq[2]["status"] == "archived"
    assert by_seq[3]["status"] == "accepted"
    assert by_seq[4]["status"] == "accepted"
    # 墓碑保留：batch_id、序号、前序与本批摘要、条数、时间、重放统计
    assert by_seq[1]["batch_id"] == "b-1"
    assert by_seq[1]["previous_digest"] == ""
    assert by_seq[2]["previous_digest"] == digests[0]
    assert by_seq[1]["digest"] == digests[0]
    assert by_seq[2]["digest"] == digests[1]
    assert by_seq[1]["observation_count"] == 3
    assert by_seq[1]["created_at"] is not None
    assert by_seq[1]["replay_count"] == 0
    assert by_seq[1]["last_replayed_at"] is None

    # 详情：墓碑的观测与阈值已清空（None），其余保留；响应带 archived_through
    d1 = client.get(f"/field-devices/{DEVICE}/batches/1").json()
    assert d1["status"] == "archived"
    assert d1["observations"] is None
    assert d1["thresholds"] is None
    assert d1["observation_count"] == 3
    assert d1["digest"] == digests[0]
    assert d1["archived_through"] == 2
    # 未归档批次详情不受影响
    d3 = client.get(f"/field-devices/{DEVICE}/batches/3").json()
    assert d3["status"] == "accepted"
    assert d3["observations"] == obs(3)
    assert d3["thresholds"] == THRESHOLDS
    assert d3["archived_through"] == 2

    # 归档状态：archived_through 随归档推进
    st = status(client, DEVICE)
    assert st["archived_through"] == 2
    assert st["safe_sequence"] == 4
    assert st["head_sequence"] == 4

    # 链头摘要不变：游标与首次受理响应保持原样
    cur = client.get(f"/field-devices/{DEVICE}/cursor").json()
    assert cur["last_sequence"] == 4
    assert cur["head_digest"] == digests[3]
    assert cur["batch_count"] == 4


def test_archive_progressive_and_count_only_new(client):
    digests, _ = upload_chain(client, DEVICE, 4)
    register(client, DEVICE, "archiver")
    ack_all(client, DEVICE, "archiver", digests, upto=2)
    r = archive(client, DEVICE, 2)
    assert r.status_code == 201 and r.json()["archived_count"] == 2

    # 继续确认到 4 后再归档：只新增 3、4 两批墓碑
    ack_all(client, DEVICE, "archiver", digests)
    r = archive(client, DEVICE, 4)
    assert r.status_code == 201
    assert r.json()["archived_count"] == 2
    assert r.json()["archived_through"] == 4
    st = status(client, DEVICE)
    assert st["archived_through"] == 4
    lst = client.get(f"/field-devices/{DEVICE}/batches").json()
    assert all(b["status"] == "archived" for b in lst["batches"])


# ---------------- 越界 409：带当前水位与阻塞消费者 ----------------


def test_archive_beyond_safe_sequence_conflict(client):
    digests, _ = upload_chain(client, DEVICE, 3)
    register(client, DEVICE, "archiver")
    register(client, DEVICE, "ml-pipeline")
    ack_all(client, DEVICE, "archiver", digests)
    ack(client, DEVICE, "ml-pipeline", 1, digests[0])
    # safe_sequence = 1（ml-pipeline 水位 1）

    r = archive(client, DEVICE, 3)
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["error"] == "archive_conflict"
    assert detail["reason"] == "beyond_safe_sequence"
    assert detail["through_sequence"] == 3
    assert detail["safe_sequence"] == 1
    assert detail["blocking_consumers"] == ["ml-pipeline"]

    # 越过安全水位即 409（即使只超 1）；未确认到目标的消费者全部列出
    r = archive(client, DEVICE, 2)
    assert r.status_code == 409
    assert r.json()["detail"]["blocking_consumers"] == ["ml-pipeline"]

    # 冲突不写动：归档水位仍为 0，批次明细完好
    assert status(client, DEVICE)["archived_through"] == 0
    d1 = client.get(f"/field-devices/{DEVICE}/batches/1").json()
    assert d1["status"] == "accepted" and d1["observations"] == obs(3)

    # 水位内的目标可以归档
    r = archive(client, DEVICE, 1)
    assert r.status_code == 201 and r.json()["archived_through"] == 1


def test_archive_without_consumers_conflicts(client):
    upload_chain(client, DEVICE, 2)
    r = archive(client, DEVICE, 1)
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["reason"] == "beyond_safe_sequence"
    assert detail["safe_sequence"] == 0
    assert detail["blocking_consumers"] == []


def test_archive_empty_device_conflicts(client):
    r = archive(client, "dev-ghost", 1)
    assert r.status_code == 409
    assert r.json()["detail"]["safe_sequence"] == 0


def test_archive_validation(client):
    assert archive(client, DEVICE, 0).status_code == 422
    assert archive(client, DEVICE, -1).status_code == 422
    r = client.post(f"/field-devices/{DEVICE}/archive", json={})
    assert r.status_code == 422


# ---------------- 幂等：重复或较小目标 ----------------


def test_archive_idempotent_repeated_and_smaller_target(client):
    digests, _ = upload_chain(client, DEVICE, 3)
    register(client, DEVICE, "archiver")
    ack_all(client, DEVICE, "archiver", digests)

    r = archive(client, DEVICE, 2)
    assert r.status_code == 201 and r.json()["status"] == "archived"

    # 重复同一目标：200 already_archived，不写动
    r = archive(client, DEVICE, 2)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "already_archived"
    assert body["archived_through"] == 2
    assert body["archived_count"] == 0

    # 较小目标同样幂等：归档水位不回退
    r = archive(client, DEVICE, 1)
    assert r.status_code == 200
    assert r.json()["status"] == "already_archived"
    assert r.json()["archived_through"] == 2

    st = status(client, DEVICE)
    assert st["archived_through"] == 2
    lst = client.get(f"/field-devices/{DEVICE}/batches").json()
    assert [b["status"] for b in lst["batches"]] == [
        "archived", "archived", "accepted",
    ]


# ---------------- 链头与后续上传不变 ----------------


def test_archive_to_head_then_upload_continues_chain(client):
    digests, _ = upload_chain(client, DEVICE, 2)
    register(client, DEVICE, "archiver")
    ack_all(client, DEVICE, "archiver", digests)
    r = archive(client, DEVICE, 2)  # 归档到链头
    assert r.status_code == 201

    # 后续上传不变：仍按链头摘要续链
    r = client.post(
        f"/field-devices/{DEVICE}/batches",
        json=batch("b-3", 3, previous_digest=digests[1]),
    )
    assert r.status_code == 201
    d3 = r.json()["digest"]
    cur = client.get(f"/field-devices/{DEVICE}/cursor").json()
    assert cur["last_sequence"] == 3 and cur["head_digest"] == d3
    assert cur["batch_count"] == 3

    # 新批次未归档，旧前缀保持墓碑
    lst = client.get(f"/field-devices/{DEVICE}/batches").json()
    assert lst["archived_through"] == 2
    assert [b["status"] for b in lst["batches"]] == [
        "archived", "archived", "accepted",
    ]
    # 归档水位不阻碍新的确认与再归档
    ack(client, DEVICE, "archiver", 3, d3)
    r = archive(client, DEVICE, 3)
    assert r.status_code == 201 and r.json()["archived_through"] == 3


# ---------------- 已归档批次：重放与确认 ----------------


def test_archived_batch_replay_counts_and_conflict(client):
    digests, responses = upload_chain(client, DEVICE, 2)
    register(client, DEVICE, "archiver")
    ack_all(client, DEVICE, "archiver", digests)
    archive(client, DEVICE, 1)

    # 同载荷重放已归档批次：按摘要接受，返回原响应（status=replayed），计数累加
    body1 = batch("b-1", 1)
    r = client.post(f"/field-devices/{DEVICE}/batches", json=body1)
    assert r.status_code == 200
    replayed = r.json()
    assert replayed["status"] == "replayed"
    assert replayed["digest"] == digests[0]
    assert {k: v for k, v in replayed.items() if k != "status"} == {
        k: v for k, v in responses[0].items() if k != "status"
    }
    r = client.post(f"/field-devices/{DEVICE}/batches", json=body1)
    assert r.status_code == 200
    d1 = client.get(f"/field-devices/{DEVICE}/batches/1").json()
    assert d1["status"] == "archived"  # 墓碑状态不被重放改写
    assert d1["replay_count"] == 2
    assert d1["last_replayed_at"] is not None

    # 链头与归档水位不因重放变化
    cur = client.get(f"/field-devices/{DEVICE}/cursor").json()
    assert cur["last_sequence"] == 2 and cur["batch_count"] == 2
    assert cur["last_event_status"] == "replayed"
    assert cur["last_event_sequence"] == 1
    assert status(client, DEVICE)["archived_through"] == 1

    # 载荷变化仍 409：同 batch_id 不同载荷 / 同序号不同 batch_id
    r = client.post(
        f"/field-devices/{DEVICE}/batches",
        json=batch("b-1", 1, observations=obs(4)),
    )
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["reason"] == "payload_changed"
    assert detail["existing_batch"] == {
        "sequence": 1, "batch_id": "b-1", "digest": digests[0],
    }
    r = client.post(
        f"/field-devices/{DEVICE}/batches", json=batch("b-x", 1)
    )
    assert r.status_code == 409
    assert r.json()["detail"]["reason"] == "payload_changed"


def test_archived_batch_ack_idempotent_no_regression(client):
    digests, _ = upload_chain(client, DEVICE, 2)
    register(client, DEVICE, "archiver")
    ack_all(client, DEVICE, "archiver", digests)
    archive(client, DEVICE, 2)

    # 正确摘要的确认幂等：已确认过的为 duplicate，水位不倒退
    r = ack(client, DEVICE, "archiver", 1, digests[0])
    assert r.status_code == 200 and r.json()["status"] == "duplicate"
    assert r.json()["watermark"] == 2
    assert r.json()["safe_sequence"] == 2
    st = status(client, DEVICE)
    assert st["consumers"][0]["watermark"] == 2
    assert st["safe_sequence"] == 2
    assert st["archived_through"] == 2

    # 已归档批次摘要冲突仍 409，附保留的期望摘要
    r = ack(client, DEVICE, "archiver", 1, "00" * 32)
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["reason"] == "digest_mismatch"
    assert detail["expected_digest"] == digests[0]


def test_new_consumer_can_ack_archived_batch_by_digest(client):
    """归档后登记的消费者补确认已归档批次：按摘要接受，水位不倒退。"""
    digests, _ = upload_chain(client, DEVICE, 2)
    register(client, DEVICE, "early")
    ack_all(client, DEVICE, "early", digests)
    archive(client, DEVICE, 2)

    r = register(client, DEVICE, "late")
    assert r.status_code == 201 and r.json()["watermark"] == 2
    # 新消费者首次确认已归档批次（无既有确认记录）：201 confirmed
    r = ack(client, DEVICE, "late", 1, digests[0])
    assert r.status_code == 201 and r.json()["status"] == "confirmed"
    assert r.json()["watermark"] == 2  # 不倒退也不越过归档水位
    # 重复确认幂等
    r = ack(client, DEVICE, "late", 1, digests[0])
    assert r.status_code == 200 and r.json()["status"] == "duplicate"
    assert status(client, DEVICE)["safe_sequence"] == 2


# ---------------- 归档后登记的新消费者 ----------------


def test_new_consumer_starts_at_archived_through(client):
    digests, _ = upload_chain(client, DEVICE, 3)
    register(client, DEVICE, "early")
    ack_all(client, DEVICE, "early", digests)
    archive(client, DEVICE, 2)

    # 归档后登记：从 archived_through 起步，首个缺口指向下一批
    r = register(client, DEVICE, "late")
    assert r.status_code == 201
    assert r.json()["watermark"] == 2
    st = status(client, DEVICE)
    assert st["archived_through"] == 2
    # safe_sequence 不被新消费者拉低到归档水位之下
    assert st["safe_sequence"] == 2
    late = [c for c in st["consumers"] if c["consumer_id"] == "late"][0]
    assert late["watermark"] == 2
    assert late["first_gap"] == 3  # 首个缺口指向下一批
    assert st["blocking_consumers"] == ["late"]

    # 重复登记幂等：水位不变
    r = register(client, DEVICE, "late")
    assert r.status_code == 200 and r.json()["watermark"] == 2

    # 新消费者确认下一批后水位推进，可继续归档
    ack(client, DEVICE, "late", 3, digests[2])
    st = status(client, DEVICE)
    assert st["safe_sequence"] == 3
    assert st["blocking_consumers"] == []
    r = archive(client, DEVICE, 3)
    assert r.status_code == 201 and r.json()["archived_through"] == 3


def test_register_between_archives_uses_latest_watermark(client):
    """归档推进后再登记的消费者从最新 archived_through 起步。"""
    digests, _ = upload_chain(client, DEVICE, 4)
    register(client, DEVICE, "a")
    ack_all(client, DEVICE, "a", digests, upto=2)
    archive(client, DEVICE, 2)
    r = register(client, DEVICE, "b")
    assert r.json()["watermark"] == 2

    ack_all(client, DEVICE, "a", digests)
    ack_all(client, DEVICE, "b", digests)
    archive(client, DEVICE, 4)
    r = register(client, DEVICE, "c")
    assert r.json()["watermark"] == 4
    st = status(client, DEVICE)
    assert st["safe_sequence"] == 4
    c = [x for x in st["consumers"] if x["consumer_id"] == "c"][0]
    assert c["first_gap"] is None  # 已追平链头


# ---------------- 并发：归档/确认/登记不越过未确认批次、水位不回退 ----------------


def test_concurrent_archive_same_target_idempotent(client):
    """并发归档同一目标：恰好一次推进，其余幂等；归档水位只增不减。"""
    n = 10
    digests, _ = upload_chain(client, DEVICE, n)
    register(client, DEVICE, "archiver")
    ack_all(client, DEVICE, "archiver", digests)

    n_threads = 8
    barrier = threading.Barrier(n_threads)
    results = []
    lock = threading.Lock()

    def worker():
        barrier.wait()
        r = archive(client, DEVICE, n)
        with lock:
            results.append((r.status_code, r.json()["status"]))

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count((201, "archived")) == 1
    assert results.count((200, "already_archived")) == n_threads - 1
    st = status(client, DEVICE)
    assert st["archived_through"] == n
    lst = client.get(f"/field-devices/{DEVICE}/batches").json()
    assert all(b["status"] == "archived" for b in lst["batches"])


def test_concurrent_archive_and_acks_never_skip_unconfirmed(client):
    """归档与确认并发：归档不得越过未确认批次——成功归档的目标必然已被
    全部必需消费者确认（归档水位恒不超过最终安全水位）。"""
    n = 20
    digests, _ = upload_chain(client, DEVICE, n)
    for cid in ("archiver", "ml"):
        register(client, DEVICE, cid)

    barrier = threading.Barrier(2 * n + 1)
    archived_ok = []
    errors = []
    lock = threading.Lock()

    def confirm(cid, seq):
        barrier.wait()
        r = ack(client, DEVICE, cid, seq, digests[seq - 1])
        if r.status_code not in (200, 201):
            with lock:
                errors.append((cid, seq, r.status_code))

    def keep_archiving():
        barrier.wait()
        # 反复尝试归档到链头：越界时 409，最终全部确认后必然成功
        for _ in range(50):
            r = archive(client, DEVICE, n)
            if r.status_code == 201:
                with lock:
                    archived_ok.append(r.json()["archived_through"])
                return
            if r.status_code != 409:
                with lock:
                    errors.append(("archive", n, r.status_code))
                return
        with lock:
            errors.append(("archive", n, "never succeeded"))

    threads = [
        threading.Thread(target=confirm, args=(cid, seq))
        for cid in ("archiver", "ml")
        for seq in range(1, n + 1)
    ]
    threads.append(threading.Thread(target=keep_archiving))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert archived_ok == [n]
    st = status(client, DEVICE)
    assert st["archived_through"] == n
    assert st["safe_sequence"] == n  # 水位未回退
    lst = client.get(f"/field-devices/{DEVICE}/batches").json()
    assert all(b["status"] == "archived" for b in lst["batches"])


def test_concurrent_register_and_archive_consistent_start(client):
    """登记与归档并发：新消费者要么挡住归档（先登记），要么从归档水位起步
    （后登记）——两种交错下 safe_sequence 都不低于 archived_through。"""
    n = 5
    digests, _ = upload_chain(client, DEVICE, n)
    register(client, DEVICE, "early")
    ack_all(client, DEVICE, "early", digests)

    barrier = threading.Barrier(2)

    def do_archive():
        barrier.wait()
        archive(client, DEVICE, n)

    def do_register():
        barrier.wait()
        register(client, DEVICE, "late")

    t1 = threading.Thread(target=do_archive)
    t2 = threading.Thread(target=do_register)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    st = status(client, DEVICE)
    late = [c for c in st["consumers"] if c["consumer_id"] == "late"][0]
    # 若归档先提交：late 从 n 起步，归档成功；若登记先提交：late 水位 0，
    # 归档被挡（archived_through=0）。两种结果都必须满足水位不变式。
    assert st["archived_through"] in (0, n)
    assert late["watermark"] == st["archived_through"]
    assert st["safe_sequence"] >= st["archived_through"]
    if st["archived_through"] == 0:
        # 归档被新消费者挡住：补齐确认后可归档
        ack_all(client, DEVICE, "late", digests)
        r = archive(client, DEVICE, n)
        assert r.status_code == 201
        assert status(client, DEVICE)["archived_through"] == n


# ---------------- 旧库迁移 ----------------


def test_legacy_device_db_gains_archived_through(tmp_path):
    """旧库（无 archived_through 列）启动后自动补列（回填 0），归档全流程可用。"""
    import json as _json
    import sqlite3

    db = tmp_path / "legacy-archive.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE device_chains (
            device_id     TEXT PRIMARY KEY,
            last_sequence INTEGER NOT NULL,
            head_digest   TEXT NOT NULL,
            batch_count   INTEGER NOT NULL,
            updated_at    REAL NOT NULL
        );
        CREATE TABLE device_batches (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id         TEXT NOT NULL,
            sequence          INTEGER NOT NULL,
            batch_id          TEXT NOT NULL,
            previous_digest   TEXT NOT NULL,
            digest            TEXT NOT NULL,
            observation_count INTEGER NOT NULL,
            thresholds        TEXT NOT NULL,
            observations      TEXT NOT NULL,
            response          TEXT NOT NULL,
            status            TEXT NOT NULL DEFAULT 'accepted',
            created_at        REAL NOT NULL,
            UNIQUE (device_id, sequence),
            UNIQUE (device_id, batch_id)
        );
        """
    )
    # 造一条旧库批次（摘要规则与现网一致即可，这里直接占位）
    d1 = "aa" * 32
    response1 = {
        "status": "accepted", "device_id": DEVICE, "batch_id": "b-1",
        "sequence": 1, "digest": d1, "previous_digest": "",
        "observation_count": 1, "received_at": "2023-11-14T22:13:20+00:00",
        "chain": {"head_sequence": 1, "head_digest": d1,
                  "next_sequence": 2, "expected_previous_digest": d1},
    }
    conn.execute(
        "INSERT INTO device_batches "
        "(device_id, sequence, batch_id, previous_digest, digest, observation_count, "
        "thresholds, observations, response, status, created_at) "
        "VALUES (?, 1, 'b-1', '', ?, 1, ?, ?, ?, 'accepted', ?)",
        (
            DEVICE, d1,
            _json.dumps(THRESHOLDS),
            _json.dumps([{"local_id": "o000", "captured_at": 1_700_000_000.0}]),
            _json.dumps(response1),
            1_700_000_000.0,
        ),
    )
    conn.execute(
        "INSERT INTO device_chains "
        "(device_id, last_sequence, head_digest, batch_count, updated_at) "
        "VALUES (?, 1, ?, 1, ?)",
        (DEVICE, d1, 1_700_000_000.0),
    )
    conn.commit()
    conn.close()

    app = create_app(db_path=str(db))
    with TestClient(app) as c:
        # 迁移回填：旧设备尚未归档任何批次
        st = status(c, DEVICE)
        assert st["archived_through"] == 0
        lst = c.get(f"/field-devices/{DEVICE}/batches").json()
        assert lst["archived_through"] == 0
        assert lst["batches"][0]["status"] == "accepted"

        # 旧库上走完整归档流程：登记 → 确认 → 归档 → 墓碑
        register(c, DEVICE, "archiver")
        ack(c, DEVICE, "archiver", 1, d1)
        r = archive(c, DEVICE, 1)
        assert r.status_code == 201 and r.json()["archived_through"] == 1
        d = c.get(f"/field-devices/{DEVICE}/batches/1").json()
        assert d["status"] == "archived"
        assert d["observations"] is None and d["thresholds"] is None
        assert d["digest"] == d1
        assert status(c, DEVICE)["archived_through"] == 1
