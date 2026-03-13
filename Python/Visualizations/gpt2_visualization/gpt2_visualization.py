from manim import *

# -------------------------------------------------------
# SCENE 1 — GPT-2 FULL ARCHITECTURE
# -------------------------------------------------------

class GPT2Architecture(Scene):
    def construct(self):

        title = Text("GPT-2 Small Architecture", font_size=56)
        title.to_edge(UP)

        self.play(Write(title))

        # Embeddings
        token_box = Rectangle(width=4, height=0.8, color=BLUE)
        token_text = Text("Token Embedding\n(50257 × 768)", font_size=26)

        pos_box = Rectangle(width=4, height=0.8, color=BLUE)
        pos_text = Text("Position Embedding\n(1024 × 768)", font_size=26)

        token = VGroup(token_box, token_text).arrange(DOWN, buff=0.15)
        pos = VGroup(pos_box, pos_text).arrange(DOWN, buff=0.15)

        embed = VGroup(token, pos).arrange(DOWN, buff=0.6)
        embed.move_to(UP * 1.5)

        self.play(Create(embed))

        # Transformer blocks
        blocks = VGroup()

        for i in range(12):
            box = Rectangle(width=4, height=0.4, color=GREEN)
            text = Text(f"Transformer Block {i}", font_size=20)
            block = VGroup(box, text).arrange(DOWN, buff=0.05)
            blocks.add(block)

        blocks.arrange(DOWN, buff=0.15)
        blocks.next_to(embed, DOWN, buff=1)

        self.play(Create(blocks))

        label = Text("12 Transformer Layers", font_size=28)
        label.next_to(blocks, RIGHT)

        self.play(Write(label))

        # LM head
        lm_box = Rectangle(width=4, height=0.8, color=ORANGE)
        lm_text = Text("LM Head\n(768 → 50257)", font_size=26)

        lm = VGroup(lm_box, lm_text).arrange(DOWN, buff=0.15)
        lm.next_to(blocks, DOWN, buff=1)

        self.play(Create(lm))

        self.wait(2)


# -------------------------------------------------------
# SCENE 2 — EXPLODED TRANSFORMER BLOCK
# -------------------------------------------------------

class TransformerExplosion(Scene):

    def construct(self):

        title = Text("Transformer Block Structure", font_size=56)
        title.to_edge(UP)

        self.play(Write(title))

        attn_box = Rectangle(width=4, height=1, color=BLUE)
        attn_text = Text("Multi-Head Attention", font_size=28)

        attn = VGroup(attn_box, attn_text).arrange(DOWN)

        mlp_box = Rectangle(width=4, height=1, color=PURPLE)
        mlp_text = Text("MLP", font_size=28)

        mlp = VGroup(mlp_box, mlp_text).arrange(DOWN)

        block = VGroup(attn, mlp).arrange(DOWN, buff=1)

        self.play(Create(block))

        arrow = Arrow(attn.get_bottom(), mlp.get_top())

        self.play(GrowArrow(arrow))

        self.wait(2)


# -------------------------------------------------------
# SCENE 3 — MLP INTERNAL STRUCTURE
# -------------------------------------------------------

class MLPStructure(Scene):

    def construct(self):

        title = Text("MLP Layer Internals", font_size=56)
        title.to_edge(UP)

        self.play(Write(title))

        fc = Rectangle(width=4, height=1, color=BLUE)
        fc_text = Text("c_fc\n768 → 3072", font_size=28)

        proj = Rectangle(width=4, height=1, color=BLUE)
        proj_text = Text("c_proj\n3072 → 768", font_size=28)

        fc_layer = VGroup(fc, fc_text).arrange(DOWN)
        proj_layer = VGroup(proj, proj_text).arrange(DOWN)

        layers = VGroup(fc_layer, proj_layer).arrange(RIGHT, buff=2)

        self.play(Create(layers))

        arrow = Arrow(fc_layer.get_right(), proj_layer.get_left())

        self.play(GrowArrow(arrow))

        self.wait(2)


# -------------------------------------------------------
# SCENE 4 — MATRIX DIMENSION VISUALIZATION
# -------------------------------------------------------

class MatrixVisualization(Scene):

    def construct(self):

        title = Text("Weight Matrix Dimensions", font_size=56)
        title.to_edge(UP)

        self.play(Write(title))

        matrix = Matrix(
            [["768", "..."],
             ["...", "3072"]],
        )

        label = Text("c_fc Weight Matrix", font_size=32)

        group = VGroup(label, matrix).arrange(DOWN)

        self.play(Write(group))

        self.wait(2)


# -------------------------------------------------------
# SCENE 5 — SVD COMPRESSION
# -------------------------------------------------------

class SVDCompression(Scene):

    def construct(self):

        title = Text("SVD Compression", font_size=56)
        title.to_edge(UP)

        self.play(Write(title))

        original = Rectangle(width=4, height=1, color=BLUE)
        original_text = Text("Weight Matrix W", font_size=28)

        original_group = VGroup(original, original_text).arrange(DOWN)

        self.play(Create(original_group))

        self.wait()

        u = Rectangle(width=2, height=0.8, color=RED)
        s = Rectangle(width=2, height=0.8, color=RED)
        v = Rectangle(width=2, height=0.8, color=RED)

        u_text = Text("U", font_size=26)
        s_text = Text("Σ", font_size=26)
        v_text = Text("Vᵀ", font_size=26)

        u_g = VGroup(u, u_text).arrange(DOWN)
        s_g = VGroup(s, s_text).arrange(DOWN)
        v_g = VGroup(v, v_text).arrange(DOWN)

        svd = VGroup(u_g, s_g, v_g).arrange(RIGHT, buff=1)

        self.play(Transform(original_group, svd))

        self.wait(2)


# -------------------------------------------------------
# SCENE 6 — KRONECKER FACTORIZATION
# -------------------------------------------------------

class KroneckerVisualization(Scene):

    def construct(self):

        title = Text("Kronecker Factorization", font_size=56)
        title.to_edge(UP)

        self.play(Write(title))

        A = Matrix([["a11","a12"],["a21","a22"]])
        B = Matrix([["b11","b12"],["b21","b22"]])

        kron = Text("A ⊗ B", font_size=40)

        group = VGroup(A, kron, B).arrange(RIGHT, buff=1)

        self.play(Write(group))

        self.wait(3)


# -------------------------------------------------------
# SCENE 7 — BEFORE VS AFTER COMPRESSION
# -------------------------------------------------------

class CompressionComparison(Scene):

    def construct(self):

        title = Text("Model Compression Effect", font_size=56)
        title.to_edge(UP)

        self.play(Write(title))

        before = Rectangle(width=3, height=4, color=RED)
        before_text = Text("Original\n124M Params", font_size=28)

        after = Rectangle(width=2, height=3, color=GREEN)
        after_text = Text("Compressed\n~40M Params", font_size=28)

        before_group = VGroup(before, before_text).arrange(DOWN)
        after_group = VGroup(after, after_text).arrange(DOWN)

        compare = VGroup(before_group, after_group).arrange(RIGHT, buff=3)

        self.play(Create(compare))

        arrow = Arrow(before_group.get_right(), after_group.get_left())

        self.play(GrowArrow(arrow))

        self.wait(3)