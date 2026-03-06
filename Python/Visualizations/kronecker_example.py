from manim import *

class KroneckerCompressionExample(Scene):

    def construct(self):

        ##################################################
        # TITLE
        ##################################################

        title = Text(
            "Matrix Compression via Kronecker Product",
            font_size=42
        )

        self.play(Write(title))
        self.wait(2)
        self.play(FadeOut(title))

        ##################################################
        # STEP 1 — ORIGINAL MATRIX
        ##################################################

        step1 = Text(
            "Step 1: Original Weight Matrix W",
            font_size=34
        ).to_edge(UP)

        W = Matrix([
            [1,2,2,4],
            [3,4,6,8],
            [5,10,1,2],
            [15,20,3,4]
        ])

        W_label = MathTex("W")

        W_group = VGroup(W_label, W).arrange(DOWN)

        self.play(Write(step1))
        self.play(Create(W_group))

        self.wait(3)

        ##################################################
        # STEP 2 — PARTITION INTO BLOCKS
        ##################################################

        step2 = Text(
            "Step 2: Partition W into 2×2 blocks",
            font_size=34
        ).to_edge(UP)

        self.play(Transform(step1, step2))

        entries = W.get_entries()

        block1 = SurroundingRectangle(
            VGroup(entries[0],entries[1],entries[4],entries[5]),
            color=BLUE
        )

        block2 = SurroundingRectangle(
            VGroup(entries[2],entries[3],entries[6],entries[7]),
            color=GREEN
        )

        block3 = SurroundingRectangle(
            VGroup(entries[8],entries[9],entries[12],entries[13]),
            color=RED
        )

        block4 = SurroundingRectangle(
            VGroup(entries[10],entries[11],entries[14],entries[15]),
            color=YELLOW
        )

        self.play(Create(block1))
        self.play(Create(block2))
        self.play(Create(block3))
        self.play(Create(block4))

        self.wait(3)

        ##################################################
        # STEP 3 — SHOW SUB BLOCKS
        ##################################################

        step3 = Text(
            "Step 3: Extract Sub-blocks",
            font_size=34
        ).to_edge(UP)

        self.play(Transform(step1, step3))

        W11 = Matrix([[1,2],[3,4]])
        W21 = Matrix([[5,10],[15,20]])
        W12 = Matrix([[2,4],[6,8]])
        W22 = Matrix([[1,2],[3,4]])

        blocks = VGroup(W11,W21,W12,W22).arrange(RIGHT,buff=1.5)

        self.play(
            FadeOut(W_group),
            FadeOut(block1),
            FadeOut(block2),
            FadeOut(block3),
            FadeOut(block4)
        )

        self.play(Create(blocks))

        self.wait(3)

        ##################################################
        # STEP 4 — VECTORIZE BLOCKS
        ##################################################

        step4 = Text(
            "Step 4: Vectorize Each Block",
            font_size=34
        ).to_edge(UP)

        self.play(Transform(step1, step4))

        v1 = Matrix([[1,2,3,4]])
        v2 = Matrix([[5,10,15,20]])
        v3 = Matrix([[2,4,6,8]])
        v4 = Matrix([[1,2,3,4]])

        vectors = VGroup(v1,v2,v3,v4).arrange(DOWN,buff=0.7)

        self.play(Transform(blocks,vectors))

        self.wait(3)

        ##################################################
        # STEP 5 — BUILD W TILDE
        ##################################################

        step5 = Text(
            "Step 5: Build Rearranged Matrix W~",
            font_size=34
        ).to_edge(UP)

        self.play(Transform(step1,step5))

        Wtilde = Matrix([
            [1,2,3,4],
            [5,10,15,20],
            [2,4,6,8],
            [1,2,3,4]
        ])

        Wt_label = MathTex(r"\tilde{W}")

        Wt_group = VGroup(Wt_label,Wtilde).arrange(DOWN)

        self.play(Transform(vectors,Wt_group))

        self.wait(3)

        ##################################################
        # STEP 6 — SHOW RANK-1 STRUCTURE
        ##################################################

        step6 = Text(
            "Step 6: Rank-1 Structure",
            font_size=34
        ).to_edge(UP)

        self.play(Transform(step1,step6))

        pattern = Tex(
            "Rows are scaled versions of [1,2,3,4]",
            font_size=30
        ).next_to(Wt_group,DOWN)

        self.play(Write(pattern))
        self.wait(3)

        ##################################################
        # STEP 7 — EXTRACT SVD VECTORS
        ##################################################

        step7 = Text(
            "Step 7: Extract Vectors u and v",
            font_size=34
        ).to_edge(UP)

        self.play(Transform(step1,step7))

        u = Matrix([[1],[5],[2],[1]])
        v = Matrix([[1],[2],[3],[4]])

        u_label = MathTex("u")
        v_label = MathTex("v")

        u_group = VGroup(u_label,u).arrange(DOWN)
        v_group = VGroup(v_label,v).arrange(DOWN)

        uv = VGroup(u_group,v_group).arrange(RIGHT,buff=2)

        self.play(
            FadeOut(Wt_group),
            FadeOut(pattern)
        )

        self.play(Create(uv))

        self.wait(3)

        ##################################################
        # STEP 8 — RESHAPE INTO A AND B
        ##################################################

        step8 = Text(
            "Step 8: Reshape into A and B",
            font_size=34
        ).to_edge(UP)

        self.play(Transform(step1,step8))

        A = Matrix([[1,2],[5,1]])
        B = Matrix([[1,2],[3,4]])

        A_label = MathTex("A")
        B_label = MathTex("B")

        A_group = VGroup(A_label,A).arrange(DOWN)
        B_group = VGroup(B_label,B).arrange(DOWN)

        AB = VGroup(A_group,B_group).arrange(RIGHT,buff=2)

        self.play(Transform(uv,AB))

        self.wait(3)

        ##################################################
        # STEP 9 — RECONSTRUCTION
        ##################################################

        step9 = Text(
            "Step 9: Reconstruct W = A ⊗ B",
            font_size=34
        ).to_edge(UP)

        self.play(Transform(step1,step9))

        W_recon = Matrix([
            [1,2,2,4],
            [3,4,6,8],
            [5,10,1,2],
            [15,20,3,4]
        ])

        kron_label = MathTex("A \\otimes B")

        kron_group = VGroup(kron_label,W_recon).arrange(DOWN)

        self.play(Transform(uv,kron_group))

        self.wait(3)

        ##################################################
        # FINAL MESSAGE
        ##################################################

        final = Text(
            "Perfect Reconstruction — Compression Achieved!",
            font_size=34
        )

        self.play(Write(final))
        self.wait(4)