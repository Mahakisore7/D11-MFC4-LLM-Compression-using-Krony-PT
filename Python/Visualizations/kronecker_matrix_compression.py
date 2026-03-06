from manim import *

class MatrixCompressionExample(Scene):

    def construct(self):

        title = Text("Matrix Compression via Kronecker Product", font_size=40)
        subtitle = Text("Step-by-Step Example", font_size=28)

        subtitle.next_to(title, DOWN)

        self.play(Write(title))
        self.play(FadeIn(subtitle))
        self.wait(2)
        self.play(FadeOut(title), FadeOut(subtitle))

        ########################################################
        # STEP 1 — Original Matrix W
        ########################################################

        step1 = Text("Step 1: Original Weight Matrix W", font_size=32).to_edge(UP)

        W = Matrix([
            [1,2,2,4],
            [3,4,6,8],
            [5,10,1,2],
            [15,20,3,4]
        ])

        W_label = MathTex("W")

        groupW = VGroup(W_label, W).arrange(DOWN)

        self.play(Write(step1))
        self.play(Create(groupW))

        self.wait(3)

        ########################################################
        # STEP 2 — Partitioning
        ########################################################

        step2 = Text("Step 2: Partition W into 2×2 blocks", font_size=32).to_edge(UP)

        self.play(Transform(step1, step2))

        block1 = SurroundingRectangle(W.get_entries()[:4], color=BLUE)
        block2 = SurroundingRectangle(W.get_entries()[4:8], color=GREEN)
        block3 = SurroundingRectangle(W.get_entries()[8:12], color=RED)
        block4 = SurroundingRectangle(W.get_entries()[12:16], color=YELLOW)

        self.play(Create(block1))
        self.play(Create(block2))
        self.play(Create(block3))
        self.play(Create(block4))

        self.wait(3)

        ########################################################
        # STEP 3 — Show Blocks
        ########################################################

        step3 = Text("Step 3: Extract Sub-blocks", font_size=32).to_edge(UP)

        self.play(Transform(step1, step3))

        W11 = Matrix([[1,2],[3,4]])
        W21 = Matrix([[5,10],[15,20]])
        W12 = Matrix([[2,4],[6,8]])
        W22 = Matrix([[1,2],[3,4]])

        blocks = VGroup(W11,W21,W12,W22).arrange(RIGHT,buff=1)

        self.play(Transform(groupW, blocks))

        self.wait(3)

        ########################################################
        # STEP 4 — Vectorization
        ########################################################

        step4 = Text("Step 4: Vectorize Each Block", font_size=32).to_edge(UP)

        self.play(Transform(step1, step4))

        v1 = Matrix([[1,2,3,4]])
        v2 = Matrix([[5,10,15,20]])
        v3 = Matrix([[2,4,6,8]])
        v4 = Matrix([[1,2,3,4]])

        vectors = VGroup(v1,v2,v3,v4).arrange(DOWN)

        self.play(Transform(blocks, vectors))

        self.wait(3)

        ########################################################
        # STEP 5 — Rearranged Matrix
        ########################################################

        step5 = Text("Step 5: Build Rearranged Matrix W~", font_size=32).to_edge(UP)

        self.play(Transform(step1, step5))

        Wtilde = Matrix([
            [1,2,3,4],
            [5,10,15,20],
            [2,4,6,8],
            [1,2,3,4]
        ])

        Wtilde_label = MathTex(r"\tilde{W}")

        group2 = VGroup(Wtilde_label,Wtilde).arrange(DOWN)

        self.play(Transform(vectors, group2))

        self.wait(3)

        ########################################################
        # STEP 6 — Rank-1 Pattern
        ########################################################

        step6 = Text("Step 6: Observe Rank-1 Pattern", font_size=32).to_edge(UP)

        self.play(Transform(step1, step6))

        pattern_text = Tex(
            "Rows are scaled versions of [1,2,3,4]",
            font_size=32
        ).next_to(group2, DOWN)

        self.play(Write(pattern_text))

        self.wait(3)

        ########################################################
        # STEP 7 — Extract Vectors
        ########################################################

        step7 = Text("Step 7: Extract SVD vectors u and v", font_size=32).to_edge(UP)

        self.play(Transform(step1, step7))

        u = Matrix([[1],[5],[2],[1]])
        v = Matrix([[1],[2],[3],[4]])

        labels = VGroup(
            MathTex("u"),
            MathTex("v")
        )

        vectors_uv = VGroup(u,v).arrange(RIGHT,buff=2)

        self.play(Transform(group2, vectors_uv))

        self.wait(3)

        ########################################################
        # STEP 8 — Reshape A and B
        ########################################################

        step8 = Text("Step 8: Reshape into matrices A and B", font_size=32).to_edge(UP)

        self.play(Transform(step1, step8))

        A = Matrix([[1,2],[5,1]])
        B = Matrix([[1,2],[3,4]])

        labels = VGroup(
            MathTex("A"),
            MathTex("B")
        )

        matrices = VGroup(
            VGroup(labels[0],A).arrange(DOWN),
            VGroup(labels[1],B).arrange(DOWN)
        ).arrange(RIGHT,buff=3)

        self.play(Transform(vectors_uv, matrices))

        self.wait(3)

        ########################################################
        # STEP 9 — Kronecker Reconstruction
        ########################################################

        step9 = Text("Step 9: Reconstruct W = A ⊗ B", font_size=32).to_edge(UP)

        self.play(Transform(step1, step9))

        kron = Matrix([
            [1,2,2,4],
            [3,4,6,8],
            [5,10,1,2],
            [15,20,3,4]
        ])

        kron_group = VGroup(
            MathTex("A \\otimes B"),
            kron
        ).arrange(DOWN)

        self.play(Transform(matrices, kron_group))

        self.wait(3)

        ########################################################
        # FINAL
        ########################################################

        final_text = Text(
            "Perfect Reconstruction — Compression Achieved!",
            font_size=34
        )

        self.play(Write(final_text))

        self.wait(4)