被审稿人打回来的一稿中联想记忆的部分：

# Associative Memory Mechanism (failed)
Timbre-agnostic transcription requires local information, while timbre separation requires global information. We posit that humans can separate timbres because they retain impressions of previously encountered timbres, enabling them to associate and categorize newly heard timbres. Hopfield network is one of the earliest associative memory networks, whose training employed Hebb's learning rule, which involves computing the node relationships for each input column vector $\mathbf{x}_i \in \{-1,+1\}^n$:
$$
\mathbf{X} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{x}_i \mathbf{x}_i^T.
$$

During inference, a vague memory $\mathbf{y} \in \mathbb{R}^n$ is input, and the matching item in memory is obtained using the following formula:
$$
\hat{\mathbf{x}} = \mathbf{y} \cdot \mathbf{X}\hspace{0.2em}.
$$

Values are then binarized to $\pm1$. Iterating this process can restore the memory. In this context, Hebb's learning rule minimizes the "network energy" of the data to be remembered (with memory points being energy minima), and the inference iterations effectively use gradient descent to gradually reduce the network energy of the input until it reaches a memory point.

Our initial idea was to train a Hopfield network during the forward propagation of a neural network to form aggregated memory points for the timbre of all time-frequency bins; then, query the most similar timbre from memory for each time-frequency bin (referred to as "association"), achieving the aggregation of similar time-frequency bin encodings and tightening the clustering.

The Hopfield network's binary nature enables perfect restoration but limits its usage. However, its concept, especially Hebb's learning rule, is worth applying to continuous values.

Assuming a time-frequency encoding matrix $V \in \mathbb{R}^{K \times D}, K=F \times T$, where $D$ is the feature dimension, $F$ is the number of frequency bins, and $T$ is the number of frames. Using Hebb's learning rule, the memory matrix $M \in \mathbb{R}^{D \times D}$ is obtained:
$$
M = \frac{V^T V}{K}.
$$

However, notes are sparse relative to the time-frequency space, and not every time-frequency bin is important; only key information should be remembered. Important time-frequency bins are those where notes are located, i.e., the frame activation result of timbre-agnostic transcription $Y_n$, which is transformed as $Y \in \mathbb{R}^{K \times 1}$ for convenience, where each element is the posterior probability of note presence, and can be used for weighting. The formula then becomes:
$$
\begin{aligned}
U \in \mathbb{R}^{N \times D} &= \frac{V}{\|V\|} \odot Y, \\
M \in \mathbb{R}^{D \times D} &= \frac{U^T U}{\mathbf{1}^T Y \mathbf{1}}.
\end{aligned}
$$

$Y$ is used to weight (rescale the amplitude of) $V$ to obtain $U$, which is then subjected to Hebb learning; the normalization numerator is subsequently modified to use the sum of the elements of $Y$. Finally, association is performed for each time-frequency bin to obtain $\hat{V} \in \mathbb{R}^{N \times D}$:
$$
\hat{V} = V \cdot M.
$$


Formula above needs to be repeated multiple times theoretically, but \cite{hopfieldisall} pointed out that one iteration can greatly approximate the memory point. The entire formula is then as follows:
$$
\hat{V} = \frac{V (U^T U)}{\mathbf{1}^T Y \mathbf{1}}.
$$

And this is precisely the attention mechanism without _Softmax_! Viewing from the perspective of attention further explains the principle of this method and is the inspiration for the implementation of "associative memory" using the attention mechanism.