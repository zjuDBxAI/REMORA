"""Training and inference helpers for full-dimensional additive residual codes."""

from __future__ import annotations

import numpy as np
import torch


def additive_reconstruction(codes: np.ndarray, codebooks: np.ndarray) -> np.ndarray:
    assignments = np.asarray(codes, dtype=np.uint8)
    centers = np.asarray(codebooks, dtype=np.float32)
    if (
        assignments.ndim != 2
        or centers.ndim != 3
        or assignments.shape[1] != centers.shape[0]
    ):
        raise ValueError("additive code/reconstruction shape mismatch")
    result = np.zeros((assignments.shape[0], centers.shape[2]), dtype=np.float32)
    for codebook in range(centers.shape[0]):
        result += centers[codebook, assignments[:, codebook]]
    return result


def conditional_context(
    primary_codes: np.ndarray, embeddings: np.ndarray
) -> np.ndarray:
    codes = np.asarray(primary_codes, dtype=np.uint8)
    tables = np.asarray(embeddings, dtype=np.float32)
    if tables.size == 0:
        return np.empty((codes.shape[0], 0), dtype=np.float32)
    if (
        codes.ndim != 2
        or tables.ndim != 3
        or codes.shape[1] != tables.shape[0]
    ):
        raise ValueError("primary-code conditional embedding shape mismatch")
    result = np.zeros((codes.shape[0], tables.shape[2]), dtype=np.float32)
    for subspace in range(codes.shape[1]):
        result += tables[subspace, codes[:, subspace]]
    return result


def conditional_prediction(
    primary_codes: np.ndarray, basis: np.ndarray, embeddings: np.ndarray
) -> np.ndarray:
    matrix = np.asarray(basis, dtype=np.float32)
    context = conditional_context(primary_codes, embeddings)
    if context.shape[1] == 0:
        return np.zeros((context.shape[0], matrix.shape[0]), dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != context.shape[1]:
        raise ValueError("conditional basis shape mismatch")
    return np.ascontiguousarray(context @ matrix.T, dtype=np.float32)


def _squared_distance(values: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    return (
        torch.sum(values * values, dim=1)[:, None]
        + torch.sum(centers * centers, dim=1)[None, :]
        - 2.0 * values @ centers.T
    )


def assign_additive_batch(
    values: np.ndarray,
    codebooks: np.ndarray,
    *,
    device: torch.device,
    coordinate_sweeps: int = 1,
    initial_codes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Greedy residual encoding followed by ICM coordinate sweeps."""
    if coordinate_sweeps < 0:
        raise ValueError("additive coordinate sweeps must be nonnegative")
    source = torch.as_tensor(np.ascontiguousarray(values, dtype=np.float32), device=device)
    centers = torch.as_tensor(np.asarray(codebooks, dtype=np.float32), device=device)
    if source.ndim != 2 or centers.ndim != 3 or source.shape[1] != centers.shape[2]:
        raise ValueError("additive assignment shape mismatch")
    rows = int(source.shape[0])
    dictionaries = int(centers.shape[0])
    reconstruction = torch.zeros_like(source)
    if initial_codes is None:
        codes = torch.zeros((rows, dictionaries), dtype=torch.long, device=device)
    else:
        initial = np.asarray(initial_codes, dtype=np.uint8)
        if initial.shape != (rows, dictionaries):
            raise ValueError("initial additive assignment shape mismatch")
        codes = torch.as_tensor(
            np.ascontiguousarray(initial), dtype=torch.long, device=device
        ).clone()
    with torch.no_grad():
        if initial_codes is None:
            for dictionary in range(dictionaries):
                target = source - reconstruction
                selected = torch.argmin(
                    _squared_distance(target, centers[dictionary]), dim=1
                )
                codes[:, dictionary] = selected
                reconstruction += centers[dictionary, selected]
        else:
            for dictionary in range(dictionaries):
                reconstruction += centers[dictionary, codes[:, dictionary]]
        for _ in range(coordinate_sweeps):
            for dictionary in range(dictionaries):
                old = centers[dictionary, codes[:, dictionary]]
                without = reconstruction - old
                target = source - without
                selected = torch.argmin(
                    _squared_distance(target, centers[dictionary]), dim=1
                )
                codes[:, dictionary] = selected
                reconstruction = without + centers[dictionary, selected]
        error = torch.sum((source - reconstruction) ** 2).item()
    return (
        codes.to("cpu").numpy().astype(np.uint8),
        reconstruction.to("cpu").numpy().astype(np.float32),
        float(error),
    )


def _batch_other_reconstruction(
    batch_codes: np.ndarray,
    codebooks: torch.Tensor,
    excluded: int,
    device: torch.device,
) -> torch.Tensor:
    codes = torch.as_tensor(
        np.ascontiguousarray(batch_codes, dtype=np.uint8),
        dtype=torch.long,
        device=device,
    )
    result = torch.zeros(
        (codes.shape[0], codebooks.shape[2]),
        dtype=codebooks.dtype,
        device=device,
    )
    for dictionary in range(int(codebooks.shape[0])):
        if dictionary != excluded:
            result += codebooks[dictionary, codes[:, dictionary]]
    return result


def _fit_additive_coordinate(
    values: np.ndarray,
    codebooks: np.ndarray,
    codes: np.ndarray,
    dictionary: int,
    iterations: int,
    *,
    device: torch.device,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    rows, dimensions = int(values.shape[0]), int(values.shape[1])
    centers_np = np.asarray(codebooks[dictionary], dtype=np.float32).copy()
    if not np.any(centers_np):
        selected = rng.choice(rows, centers_np.shape[0], replace=False)
        sample = np.asarray(values[selected], dtype=np.float32)
        other = np.zeros_like(sample)
        for other_dictionary in range(codebooks.shape[0]):
            if other_dictionary != dictionary and np.any(codebooks[other_dictionary]):
                other += codebooks[
                    other_dictionary, codes[selected, other_dictionary]
                ]
        centers_np[:] = sample - other
    history: list[float] = []
    for _ in range(iterations):
        all_centers = torch.as_tensor(
            np.asarray(codebooks, dtype=np.float32), device=device
        )
        centers = torch.as_tensor(centers_np, device=device)
        sums = torch.zeros_like(centers)
        counts = torch.zeros(centers.shape[0], dtype=torch.int64, device=device)
        distortion = 0.0
        with torch.no_grad():
            for start in range(0, rows, batch_size):
                end = min(rows, start + batch_size)
                source = torch.as_tensor(
                    np.ascontiguousarray(values[start:end], dtype=np.float32),
                    device=device,
                )
                other = _batch_other_reconstruction(
                    codes[start:end], all_centers, dictionary, device
                )
                target = source - other
                distance = _squared_distance(target, centers)
                selected = torch.argmin(distance, dim=1)
                codes[start:end, dictionary] = (
                    selected.to("cpu").numpy().astype(np.uint8)
                )
                sums.index_add_(0, selected, target)
                counts += torch.bincount(selected, minlength=centers.shape[0])
                distortion += float(
                    torch.sum((target - centers[selected]) ** 2).item()
                )
        nonempty = counts > 0
        updated = centers.clone()
        updated[nonempty] = sums[nonempty] / counts[nonempty, None]
        centers_np = updated.to("cpu").numpy().astype(np.float32)
        codebooks[dictionary] = centers_np
        history.append(distortion / max(rows, 1))
    return centers_np, counts.to("cpu").numpy().astype(np.int64), history


def additive_distortion(
    values: np.ndarray,
    codebooks: np.ndarray,
    codes: np.ndarray,
    *,
    batch_size: int,
) -> float:
    total = 0.0
    for start in range(0, values.shape[0], batch_size):
        end = min(values.shape[0], start + batch_size)
        difference = np.asarray(values[start:end], dtype=np.float32) - additive_reconstruction(
            codes[start:end], codebooks
        )
        total += float(np.sum(difference.astype(np.float64) ** 2))
    return total / max(int(values.shape[0]), 1)


def assign_additive_dataset(
    values: np.ndarray,
    codebooks: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    coordinate_sweeps: int,
    initial_codes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Encode a dataset in bounded-memory batches with fixed codebooks."""
    source = np.asarray(values)
    dictionaries = int(np.asarray(codebooks).shape[0])
    codes = np.empty((source.shape[0], dictionaries), dtype=np.uint8)
    counts = np.zeros(
        (dictionaries, np.asarray(codebooks).shape[1]), dtype=np.int64
    )
    squared_error = 0.0
    for start in range(0, source.shape[0], batch_size):
        end = min(source.shape[0], start + batch_size)
        assigned, _reconstruction, error = assign_additive_batch(
            source[start:end],
            codebooks,
            device=device,
            coordinate_sweeps=coordinate_sweeps,
            initial_codes=(
                None if initial_codes is None else initial_codes[start:end]
            ),
        )
        codes[start:end] = assigned
        squared_error += error
        for dictionary in range(dictionaries):
            counts[dictionary] += np.bincount(
                assigned[:, dictionary].astype(np.int64),
                minlength=np.asarray(codebooks).shape[1],
            )
    return codes, counts, squared_error / max(int(source.shape[0]), 1)


def train_additive_codebooks(
    values: np.ndarray,
    *,
    dictionaries: int,
    codewords: int,
    initialization_iterations: int,
    joint_sweeps: int,
    coordinate_iterations: int,
    device: torch.device,
    batch_size: int,
    seed: int,
    initial_codebooks: np.ndarray | None = None,
    initial_codes: np.ndarray | None = None,
    log_prefix: str = "additive",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    source = np.asarray(values)
    if (
        source.ndim != 2
        or source.shape[0] < codewords
        or dictionaries <= 0
        or codewords != 256
        or initialization_iterations <= 0
        or joint_sweeps < 0
        or coordinate_iterations <= 0
        or batch_size <= 0
    ):
        raise ValueError("invalid additive-quantizer training configuration")
    rng = np.random.default_rng(seed)
    if initial_codebooks is None:
        codebooks = np.zeros(
            (dictionaries, codewords, source.shape[1]), dtype=np.float32
        )
        codes = np.zeros((source.shape[0], dictionaries), dtype=np.uint8)
        stages = range(dictionaries)
        stage_iterations = initialization_iterations
    else:
        codebooks = np.asarray(initial_codebooks, dtype=np.float32).copy()
        if codebooks.shape != (dictionaries, codewords, source.shape[1]):
            raise ValueError("initial additive codebook shape mismatch")
        if initial_codes is None:
            codes = np.empty((source.shape[0], dictionaries), dtype=np.uint8)
            for start in range(0, source.shape[0], batch_size):
                end = min(source.shape[0], start + batch_size)
                codes[start:end] = assign_additive_batch(
                    source[start:end], codebooks, device=device, coordinate_sweeps=1
                )[0]
        else:
            codes = np.asarray(initial_codes, dtype=np.uint8).copy()
            if codes.shape != (source.shape[0], dictionaries):
                raise ValueError("initial additive code shape mismatch")
        stages = range(0)
        stage_iterations = 0
    counts = np.zeros((dictionaries, codewords), dtype=np.int64)
    history: list[dict] = []
    for dictionary in stages:
        _, counts[dictionary], losses = _fit_additive_coordinate(
            source,
            codebooks,
            codes,
            dictionary,
            stage_iterations,
            device=device,
            batch_size=batch_size,
            rng=rng,
        )
        distortion = additive_distortion(
            source, codebooks, codes, batch_size=batch_size
        )
        history.append(
            {"stage": "rq-initialization", "dictionary": dictionary, "losses": losses, "mse": distortion}
        )
        print(
            f"[{log_prefix}] initialized={dictionary + 1}/{dictionaries} mse={distortion:.8g}",
            flush=True,
        )
    for sweep in range(joint_sweeps):
        before = additive_distortion(source, codebooks, codes, batch_size=batch_size)
        checkpoint_codebooks = codebooks.copy()
        checkpoint_codes = codes.copy()
        checkpoint_counts = counts.copy()
        sweep_losses = []
        for dictionary in range(dictionaries):
            _, counts[dictionary], losses = _fit_additive_coordinate(
                source,
                codebooks,
                codes,
                dictionary,
                coordinate_iterations,
                device=device,
                batch_size=batch_size,
                rng=rng,
            )
            sweep_losses.append(losses)
        after = additive_distortion(source, codebooks, codes, batch_size=batch_size)
        accepted = bool(after <= before * (1.0 + 1e-7))
        if not accepted:
            codebooks = checkpoint_codebooks
            codes = checkpoint_codes
            counts = checkpoint_counts
            after = before
        history.append(
            {
                "stage": "joint-coordinate",
                "sweep": sweep + 1,
                "before_mse": before,
                "after_mse": after,
                "accepted": accepted,
                "coordinate_losses": sweep_losses,
            }
        )
        print(
            f"[{log_prefix}] sweep={sweep + 1} mse={after:.8g}", flush=True
        )
        if not accepted:
            break
    final = additive_distortion(source, codebooks, codes, batch_size=batch_size)
    diagnostics = {
        "method": "sequential_rq_initialization_plus_additive_coordinate_descent",
        "initialization_iterations": initialization_iterations,
        "joint_sweeps": joint_sweeps,
        "coordinate_iterations": coordinate_iterations,
        "final_metric_mse": final,
        "empty_sample_codewords": int(np.sum(counts == 0)),
        "history": history,
    }
    return codebooks, codes, counts, diagnostics


def fit_low_rank_primary_conditioner(
    values: np.ndarray,
    primary_codes: np.ndarray,
    side_codes: np.ndarray,
    codebooks: np.ndarray,
    *,
    rank: int,
    primary_codewords: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    batch_size: int,
    seed: int,
    initial_basis: np.ndarray | None = None,
    initial_embeddings: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    source = np.asarray(values)
    primary = np.asarray(primary_codes, dtype=np.uint8)
    side = np.asarray(side_codes, dtype=np.uint8)
    if (
        source.ndim != 2
        or primary.ndim != 2
        or side.ndim != 2
        or source.shape[0] != primary.shape[0]
        or source.shape[0] != side.shape[0]
        or rank <= 0
        or rank > source.shape[1]
        or primary_codewords <= 0
        or epochs <= 0
        or (primary.size > 0 and int(np.max(primary)) >= primary_codewords)
    ):
        raise ValueError("invalid conditional residual training configuration")
    torch.manual_seed(seed)
    initialization = "provided"
    unconditioned_metric_mse: float | None = None
    if initial_basis is None:
        covariance = np.zeros(
            (source.shape[1], source.shape[1]), dtype=np.float64
        )
        for start in range(0, source.shape[0], batch_size):
            end = min(source.shape[0], start + batch_size)
            target = np.asarray(source[start:end], dtype=np.float32) - additive_reconstruction(
                side[start:end], codebooks
            )
            target64 = target.astype(np.float64)
            covariance += target64.T @ target64
        unconditioned_metric_mse = float(
            np.trace(covariance) / max(int(source.shape[0]), 1)
        )
        _eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        basis_np = np.ascontiguousarray(
            eigenvectors[:, -rank:], dtype=np.float32
        )
        initialization = "target_pca_plus_categorical_coordinate_mean"
    else:
        basis_np = np.asarray(initial_basis, dtype=np.float32)
    if initial_embeddings is None:
        embeddings_np = np.zeros(
            (primary.shape[1], primary_codewords, rank), dtype=np.float32
        )
        projected = np.empty((source.shape[0], rank), dtype=np.float32)
        for start in range(0, source.shape[0], batch_size):
            end = min(source.shape[0], start + batch_size)
            target = np.asarray(source[start:end], dtype=np.float32) - additive_reconstruction(
                side[start:end], codebooks
            )
            projected[start:end] = target @ basis_np
        context_np = np.zeros_like(projected)
        for subspace in range(primary.shape[1]):
            labels = primary[:, subspace].astype(np.int64)
            target = projected - context_np
            counts = np.bincount(labels, minlength=primary_codewords).astype(
                np.float64
            )
            nonempty = counts > 0
            for coordinate in range(rank):
                sums = np.bincount(
                    labels,
                    weights=target[:, coordinate].astype(np.float64),
                    minlength=primary_codewords,
                )
                embeddings_np[subspace, nonempty, coordinate] = (
                    sums[nonempty] / counts[nonempty]
                ).astype(np.float32)
            context_np += embeddings_np[subspace, labels]
    else:
        embeddings_np = np.asarray(initial_embeddings, dtype=np.float32)
    if basis_np.shape != (source.shape[1], rank):
        raise ValueError("conditional basis shape mismatch")
    if embeddings_np.shape != (primary.shape[1], primary_codewords, rank):
        raise ValueError("conditional embedding shape mismatch")
    basis = torch.nn.Parameter(torch.as_tensor(basis_np, device=device))
    embeddings = torch.nn.Parameter(torch.as_tensor(embeddings_np, device=device))
    fixed_codebooks = torch.as_tensor(np.asarray(codebooks, dtype=np.float32), device=device)
    optimizer = torch.optim.Adam(
        (basis, embeddings), lr=learning_rate, weight_decay=weight_decay
    )
    rng = np.random.default_rng(seed)
    history: list[dict[str, float | int | bool]] = []
    subspace_ids = torch.arange(primary.shape[1], device=device)[None, :]

    def evaluate_metric_mse() -> float:
        squared_error = 0.0
        with torch.no_grad():
            for eval_start in range(0, source.shape[0], batch_size):
                eval_end = min(source.shape[0], eval_start + batch_size)
                batch_values = torch.as_tensor(
                    np.ascontiguousarray(
                        source[eval_start:eval_end], dtype=np.float32
                    ),
                    device=device,
                )
                batch_primary = torch.as_tensor(
                    np.ascontiguousarray(
                        primary[eval_start:eval_end], dtype=np.uint8
                    ),
                    dtype=torch.long,
                    device=device,
                )
                batch_side = torch.as_tensor(
                    np.ascontiguousarray(
                        side[eval_start:eval_end], dtype=np.uint8
                    ),
                    dtype=torch.long,
                    device=device,
                )
                side_reconstruction = torch.zeros_like(batch_values)
                for dictionary in range(side.shape[1]):
                    side_reconstruction += fixed_codebooks[
                        dictionary, batch_side[:, dictionary]
                    ]
                context = embeddings[subspace_ids, batch_primary].sum(dim=1)
                prediction = context @ basis.T
                squared_error += float(
                    torch.sum(
                        (batch_values - side_reconstruction - prediction) ** 2
                    ).item()
                )
        return squared_error / max(int(source.shape[0]), 1)

    initial_metric_mse = evaluate_metric_mse()
    best_metric_mse = initial_metric_mse
    best_epoch = 0
    best_basis = basis.detach().clone()
    best_embeddings = embeddings.detach().clone()
    for epoch in range(epochs):
        permutation = rng.permutation(source.shape[0])
        total = 0.0
        for start in range(0, source.shape[0], batch_size):
            end = min(source.shape[0], start + batch_size)
            ids = permutation[start:end]
            batch_values = torch.as_tensor(
                np.ascontiguousarray(source[ids], dtype=np.float32), device=device
            )
            batch_primary = torch.as_tensor(
                np.ascontiguousarray(primary[ids], dtype=np.uint8),
                dtype=torch.long,
                device=device,
            )
            batch_side = torch.as_tensor(
                np.ascontiguousarray(side[ids], dtype=np.uint8),
                dtype=torch.long,
                device=device,
            )
            side_reconstruction = torch.zeros_like(batch_values)
            for dictionary in range(side.shape[1]):
                side_reconstruction += fixed_codebooks[
                    dictionary, batch_side[:, dictionary]
                ]
            context = embeddings[subspace_ids, batch_primary].sum(dim=1)
            prediction = context @ basis.T
            target = batch_values - side_reconstruction
            loss = torch.mean((target - prediction) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_((basis, embeddings), 10.0)
            optimizer.step()
            total += float(loss.detach().item()) * (end - start)
        with torch.no_grad():
            q, r = torch.linalg.qr(basis, mode="reduced")
            embeddings.copy_(embeddings @ r.T)
            basis.copy_(q)
        mean_element_mse = total / max(source.shape[0], 1)
        metric_mse = evaluate_metric_mse()
        accepted = bool(metric_mse < best_metric_mse)
        if accepted:
            best_metric_mse = metric_mse
            best_epoch = epoch + 1
            best_basis = basis.detach().clone()
            best_embeddings = embeddings.detach().clone()
        history.append(
            {
                "epoch": epoch + 1,
                "mean_element_mse_before_updates": mean_element_mse,
                "metric_mse_after_epoch": metric_mse,
                "is_best": accepted,
            }
        )
        print(
            f"[primary-conditioned-residual] epoch={epoch + 1} "
            f"metric_mse={metric_mse:.8g} best={best_metric_mse:.8g}",
            flush=True,
        )
    with torch.no_grad():
        basis.copy_(best_basis)
        embeddings.copy_(best_embeddings)
    return (
        basis.detach().to("cpu").numpy().astype(np.float32),
        embeddings.detach().to("cpu").numpy().astype(np.float32),
        {
            "epochs": epochs,
            "initial_metric_mse": initial_metric_mse,
            "unconditioned_metric_mse": unconditioned_metric_mse,
            "best_metric_mse": best_metric_mse,
            "best_epoch": best_epoch,
            "iteration_zero_is_checkpoint": True,
            "initialization": initialization,
            "history": history,
            "rank": rank,
        },
    )
