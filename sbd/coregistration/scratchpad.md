Let me ground this in the references you shared and the established literature on script↔subtitle alignment before laying out a plan.

## Bottom line up front

Your DTW pipeline is a solid baseline, and it's exactly what the canonical script-alignment literature started with (Everingham et al.'s subtitle-DTW is the reference approach). But be precise about which failure mode each thing addresses, because the two DTW variants you linked solve different problems: [cvit.iiit.ac.in]


| Failure mode | Violates which DTW axiom | Fixed by a DTW variant? |
|---|---|---|
| Scene deletions (in script, not in film) | Boundary + step-size (forces full consumption of both sequences) | Partially — step-size + subsequence relaxations |
| Scene insertions / ad-libs (in film, not in script) | Same, mirrored | Partially — same relaxations |
| Scene reordering | Monotonicity | No. No DTW variant handles this |

The key thing: reordering breaks the monotonicity axiom that all DTW variants preserve. Global DTW, slope-constrained DTW, and subsequence DTW all still require the warping path to be monotonically non-decreasing in both axes. So no amount of DTW tuning will recover a transposed scene. That's the fundamental limit — and it's a known one; there's literally a paper titled*"Scene reordering in movie script alignment"* addressing exactly this gap. [ieeexplore.ieee.org]

## What the two variants you linked actually buy you

Step-size conditions (C3S2): replacing with bounds the path slope between 1/2 and 2, killing the degenerate "one line matched to 20 subtitles" horizontal/vertical runs. Crucially, this variant allows some elements to go unmatched, which is a mild, local form of skip tolerance. Good for noise robustness; not a real deletion handler. [audiolabs-...rlangen.de]

Subsequence DTW (C7S2): relaxes the boundary condition so a short query can be matched to an optimal contiguous sub-stretch of a long sequence, skipping its beginning and end for free. This is the genuinely useful one for you — but not the way you'd use it globally. Use it per-scene: treat each screenplay scene's dialogue as a short query and the full subtitle stream as the long . Because each scene is located independently, deletions and reordering fall out naturally at scene granularity. The cost: you lose global consistency (two scenes can claim overlapping subtitle ranges), which you then resolve in a chaining step. [audiolabs-...rlangen.de]

## Recommended architecture: seed → chain → refine

This is the framing I'd move to. It's the standard "anchor-based / colinear chaining" paradigm from genome alignment, which was built precisely for sequences with insertions, deletions, and rearrangements — a much better structural match to your problem than monotonic warping. [emergentmind.com], [arxiv.org]

1. Build the similarity matrix (seeds). Movie dialogue is usually near-verbatim of the screenplay, so pure semantic embeddings can over-match (different lines that are semantically similar). Use a hybrid score: normalized fuzzy text match (Levenshtein / token-set ratio) as the primary signal, embeddings as the paraphrase/ad-lib fallback. The existing open-source script↔subtitle aligners lean on exact/fuzzy string comparison for this reason, and the StoryMovie pipeline uses Longest-Common-Subsequence matching on dialogue. [github.com] [arxiv.org]

2. Extract anchors. Keep only high-confidence matches: mutual-best-match + uniqueness (a script line matching essentially one subtitle line above threshold). These are your reliable pins. Reordered scenes show up as off-diagonal diagonal segments in the similarity matrix — visualize it as a dot-plot; you'll see the transpositions.

3. Chain the anchors.

If you assume no reordering: colinear chaining (or DTW/subsequence-DTW) over anchors. Chaining beats DTW here because it handles gaps (deletions/insertions) natively instead of smearing them. [arxiv.org]
If you must handle reordering: use a jump-limited reordering model. Cour et al.'s hierarchical DP does exactly this — it aligns screenplay scene labels to the movie while allowing jump-limited reorderings in linear time. Alternatively, an HMM/CRF sequence-labeling formulation (states = script sentences) as in Everingham/Naim handles noise and non-monotonic jumps more gracefully than DP warping. [repository.upenn.edu] [cvit.iiit.ac.in], [cs.rochester.edu]
4. Refine locally. Between consecutive anchors, run your original slope-constrained DTW on the dialogue to get line-level timestamps. Then a screenplay scene's [start, end] = min/max subtitle timestamp of its matched lines, padded to cover the surrounding action lines.

## Exploit the screenplay's structure (the practical sweet spot)

Screenplays are hierarchical: sluglines (INT./EXT. LOCATION - TIME) give you free scene boundaries. So use a two-level model:

Between scenes: allow reordering/deletion (chaining or jump-limited DP).
Within a scene: assume monotonic order → your DTW works perfectly here.
This matches how the film is actually built (scenes are the reordering/cutting unit, dialogue within a take stays in order) and dramatically shrinks the reordering search space. This scene-block-as-query idea is exactly where subsequence DTW earns its place.

## Evaluation without ground truth

Since you flagged no labels, here's how to validate defensibly:

Synthetic ground truth (do this first). Take a movie where your naive DTW already aligns cleanly, then programmatically delete and permute scenes in the screenplay. You now have exact ground truth for a controlled reordering/deletion benchmark — measure recovery rate of anchors and timestamp error. This is the single most valuable move you can make.
Cross-signal agreement as pseudo-labels. Where fuzzy-text and embedding matches independently agree on the same anchor, treat it as a silver label; measure your final alignment against that set.
Internal consistency metrics (no labels needed): monotonicity of timestamps within each scene, plausibility of scene durations (flag <10s or >15min), subtitle coverage (% of lines confidently matched), and total chaining/warping cost distribution.
Targeted manual spot-checks on the lowest-confidence 5–10% of segments — that's where errors concentrate.
Verdict on "is DTW valid?"

Not to be discarded, but demote it from "the algorithm" to "the local-refinement step." Use anchor-based colinear chaining (with jump-limited reordering if needed) as the global backbone, subsequence DTW for scene-block localization, and slope-constrained DTW inside anchor gaps. That decomposition respects the one thing standard DTW can never give you — non-monotonic (reordered) alignment.

Want me to prototype the dot-plot similarity matrix + anchor extraction + a jump-limited chaining pass on a sample script/.srt pair so you can see the reordering segments and test the two-level model?

## The chaining step, concretely

After anchor extraction you have a set of anchors , each a matched pair : a screenplay line index (in script order), a subtitle line index (in movie/time order), and a confidence weight . Plot them as a dot-plot — script axis vs. time axis:

Perfectly-followed screenplay → points on the main diagonal.
Deletion / insertion → a gap (a jump along one axis with no movement on the other).
Reordering → an off-diagonal block, or a segment running against the diagonal.
"Chaining" = turning this scatter of noisy pins into one globally coherent, timestamped alignment. Every method below is really answering one question: which subset of anchors do I trust, and in what order do I string them together? They differ in the structural axioms they impose — and those axioms are exactly what determine whether deletions and reordering survive.

## The four axioms to keep in mind:

| Axiom | Meaning |
|---|---|
| Monotonicity | path never decreases on either axis → order preserved (kills reordering) |
| Completeness | every element must be matched to something (no free skips) |
| Boundary | must start at and end at |
| Gap tolerance | unmatched stretches allowed and priced, not smeared |


## The five approaches

1. Colinear chaining

Select the maximum-weight subset of anchors that is strictly increasing in both coordinates. Sort anchors, then run sparse DP with a range-max query structure (a Fenwick/segment tree over the second coordinate):


Runs in over the number of anchors — not over — so it's extremely fast and sparse. Gaps between consecutive anchors are priced by an explicit gap cost, so deletions/insertions are handled natively rather than being distorted into staircase runs. Modern formulations tie the chaining cost directly to anchored edit distance / LCS length. [arxiv.org], [emergentmind.com]

✅ Fast, sparse, gap-native, principled cost model, no training data.
⚠️ A single chain is monotonic → no reordering. The fix: extract multiple chains. Each reordered scene falls out as its own chain, and you order the blocks afterward. Also it only pins anchors — you still need a refine pass for dense per-line timestamps.
2. Global (classic) DTW

Dense DP over the full cost matrix with boundary + monotonicity + completeness. Every script line must consume subtitle mass.

✅ Dense, optimal under its constraints, needs no anchors, trivial to implement.
❌ Completeness is the killer: a deleted scene has nothing to match, so DTW smears it into horizontal/vertical runs, corrupting neighbors. Monotonic (no reorder). time/space — heavy for a feature-length film. Fragile to outliers because everything is forced to match.
3. Subsequence DTW

Relaxes the boundary condition: match a short query into the best contiguous sub-stretch of the long sequence , skipping 's beginning and end for free. This is the variant that actually earns a place in your pipeline — but used per scene-block, not globally: take one screenplay scene's dialogue as , the whole subtitle stream as , and localize it. Because each scene is placed independently, deletions and reordering are tolerated at scene granularity. [audiolabs-...rlangen.de]

✅ Localizes a block into the timeline; great for the "scene-as-query" model.
⚠️ Free skips only at the ends of , not the middle → a mid-scene cut still smears. Still monotonic within the matched region. Run independently per scene, it gives no global consistency (two scenes can claim overlapping time), so you need a reconciliation/chaining pass on top. per query.
4. Hierarchical DP (Cour et al.)

A two-level unified model: a scene-level alignment that permits jump-limited reorderings, plus a within-scene monotonic alignment, solved in linear time via a novel hierarchical DP. This is the natural "one-model" upgrade because it mirrors how films are actually built — scenes are the cutting/reordering unit, dialogue within a take stays ordered. [repository.upenn.edu]

✅ Explicitly handles reordering (bounded), exploits screenplay hierarchy (sluglines = free scene boundaries), linear time, no labels required.
⚠️ The jump limit is a hyperparameter — set it too small and you miss long-range moves; too large and cost/false-matches grow. More work to implement than chaining.
5. HMM / CRF sequence labeling

Reframe alignment as labeling: each subtitle line (or shot) is assigned a state = the script sentence it corresponds to. Transition probabilities encode the ordering prior and can permit jumps. In the movie-alignment literature a discriminative LCRF significantly outperformed generative HMMs and prior state-of-the-art. [cvit.iiit.ac.in] [cs.rochester.edu]

✅ Probabilistic and noise-robust; soft assignments; naturally fuses many cues (text, face, audio); transitions can express non-monotonic jumps with priors.
❌ The CRF is a dealbreaker for you: it's discriminative and needs labeled training data, which you don't have. An HMM is still viable unsupervised (train emissions/transitions via EM), but the per-line state space is large ( to Viterbi) and continuous reordering is awkward to encode.
Head-to-head

| Method | Monotonic? | Deletions | Reordering | Needs anchors | Needs labels | Complexity | Output |
|---|---|---|---|---|---|---|---|
| Colinear chaining | single chain yes; multi-chain no | Yes: native (gap cost) | Yes via multi-chain | Yes | No | O(nlogn) | sparse pins |
| Global DTW | Yes | No: smears | No | No | No | O(NM) | dense |
| Subsequence DTW | within region | partial (ends only) | Yes: per-block use | No | No | O(NM)/query | dense per block |
| Hierarchical DP | within scene | Yes | Yes: jump-limited | optional | No | ~linear | dense, structured |
| HMM | soft/priored | Yes | limited (via transitions) | No | No (EM) | O(TS) – O(TS^2) | soft labels |
| CRF | soft | Yes | Yes: w/ features | No | Yes | training-dependent | soft labels |


## What I'd actually pick for your constraints

Given reordering + deletions + no labels, use a layered design:

Global backbone → multi-chain colinear chaining over anchors. Cheap, gap-native, and reordered scenes surface as separate chains you then order in time. This is your primary chaining engine.
If reordering is frequent and you want one principled model → hierarchical DP with jump-limited scene reordering — it maps cleanly onto sluglines and film structure. [repository.upenn.edu]
Block localization → subsequence DTW per screenplay scene to place each block on the timeline. [audiolabs-...rlangen.de]
Refine inside anchor gaps → slope-constrained DTW for dense line-level timestamps.
Drop CRF (no labels); keep an unsupervised HMM only as an optional smoothing/robustness layer if anchor noise is high.
Rule of thumb: chaining is the right global tool because it prices gaps and permits blocks; DTW is the right local tool because it gives dense, monotonic alignment where order genuinely holds. Don't ask one to do the other's job.

Want me to prototype the multi-chain colinear chaining DP (with the segment-tree range-max and a gap cost) on a sample dot-plot so you can see reordered scenes split into distinct chains?


## LCRF

Let me pull the actual Naim et al. paper so I describe the LCRF accurately rather than from memory.Good news up front: this is where I need to refine what I told you earlier. I lumped "CRF" together as needing labels — but the Latent CRF (LCRF) in Naim et al. is specifically engineered to avoid element-level alignment labels. The alignment itself is the hidden variable. So it's much more compatible with your no-ground-truth constraint than a vanilla CRF. Let me walk through it properly.

## The setup Naim et al. actually solve

Their task: align a sequence of character-name mentions from a script to a sequence of face-track clusters from the movie, without subtitles and without hand-aligned frames. The load-bearing prior is temporal ordering: faces generally appear in the same sequential order as their names are mentioned in the script. That's structurally identical to your problem — swap "name mentions → face clusters" for "screenplay dialogue lines → subtitle lines." [cs.rochester.edu] [cs.rochester.edu]

They tried both a generative HMM and a discriminative LCRF, and the LCRF significantly outperformed the HMM and the prior state of the art. [cs.rochester.edu]

## What makes it "Latent"

A standard linear-chain CRF models where is the label sequence you want, and to train it you need pairs — i.e., you must already know the correct labels. That's the version that needs annotations.

An LCRF inserts a hidden alignment variable. Restructure the pieces:

Observations : the sequence of movie elements (face clusters; for you, subtitle lines with embeddings/timestamps).
Labels / targets : the sequence of script elements (name mentions; for you, screenplay dialogue lines). Critically, you already know — it's just the script, read in order. It is not something you annotate.
Latent variable (or ): the alignment — which observation corresponds to which script element. This is what you don't know and never labeled.
The model is discriminative over the observation-label pair, marginalizing the alignment:



So the alignment is summed out. You never supply it. This is exactly the "latent-variable discriminative model for the unsupervised alignment task" framing from their companion NAACL paper, whose entire point is aligning language to video**"without any direct supervision"** and without "hand-aligned parallel data". [cs.rochester.edu]

## Why discriminative helps here

The reason they went discriminative rather than sticking with the HMM: a CRF-family model lets you pour in diverse, arbitrary, overlapping features without worrying about them being probabilistically independent (which a generative HMM assumes). Their own words: the discriminative models "are capable of incorporating domain knowledge, by adding diverse and overlapping features," and they outperformed the generative models on alignment accuracy. [cs.rochester.edu]

## Feature functions you could define for script↔subtitle:

Emission-style: cosine similarity of line embeddings; fuzzy text-match ratio; length ratio; shared named entities/character names.
Transition-style: does aligning line just after line preserve monotonic order? penalty for backward jumps; penalty proportional to skip size (deletion cost).
Structural: slugline/scene-boundary agreement; speaker-name agreement.
Training — and the crux: what annotations it needs

Here's the payoff for your constraint. Because is latent, you train by maximizing the conditional likelihood of the known label sequence given the observations, marginalizing over all alignments:



This is a non-convex objective (latent-variable models usually are) optimized with gradient methods or an EM-style / latent structured-perceptron loop, where the ordering constraint restricts the space of admissible so inference stays tractable.

## Annotation requirements, ranked:

| What | Element-level alignment labels? | Needed here? |
|---|---|---|
| Vanilla linear-chain CRF | required (every line→line pair) | you can't provide this |
| LCRF (Naim) | not required | this is the point |
| What LCRF does need | paired sequences: one script ↔ one movie | you have this for free |

So the only "supervision" is distant / weak: the fact that this script goes with this movie. You are not annotating which dialogue line maps to which subtitle. That pairing you already have by construction. That directly overturns my earlier "dealbreaker — needs labeled training data" claim: it applies to the plain CRF, not the latent variant.

## The honest caveats

You still need paired data across multiple films to learn feature weights that generalize. For a single movie you'd be fitting a handful of weights on one sequence pair — high variance. Practically you'd train on a batch of script/subtitle pairs (all unlabeled at the line level), then apply to new films. If you only ever have one film, a hand-tuned cost function + chaining is simpler and about as good.
Non-convex training needs decent initialization (e.g., seed with your anchor matches) to avoid poor local optima.
Reordering: the vanilla LCRF encodes a monotonic ordering prior. It tolerates skips (deletions) gracefully via transition features, but large-scale scene transpositions still need either jump-permitting transitions (with a jump penalty) or the hierarchical/scene-block decomposition we discussed. The ordering assumption is baked into the prior. [cs.rochester.edu]
For evaluation you still have no labels — so you'd validate the LCRF the same way as everything else: synthetic deletion/reorder benchmarks and cross-signal agreement.
Modern alternative worth knowing

If you want the LCRF's "latent alignment, no element labels" property but with neural features, look at Neural-Hidden-CRF: it embeds a hidden CRF layer that models the word sequence, the latent ground-truth sequence, and a weak-label sequence, using BERT-style contextual features, and is explicitly built for weakly-supervised sequence labeling with no truth labels used for learning ("weak supervision zero-shot sequence labeling"). Same philosophical trick (latent truth, discriminative, rich neural features), more current tooling. [arxiv.org], [github.com]

## Where this leaves your pipeline

LCRF is viable under no ground truth — the alignment is latent; you only need script↔movie pairing, ideally several pairs to fit weights.
It's heavier than chaining and needs careful training; I'd still keep multi-chain colinear chaining as your default backbone and treat LCRF as the upgrade path when you have a corpus of films and want a learned, feature-rich model that fuses text + speaker + structural cues.
Its monotonic ordering prior means reordering still rides on transition penalties or the scene-block decomposition — not free.
Want me to sketch the LCRF factor graph for the script↔subtitle case — nodes, the specific emission/transition feature functions, and the forward-backward-with-latent-alignment training loop — so you can see exactly what you'd implement?