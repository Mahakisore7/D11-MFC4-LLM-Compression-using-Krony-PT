from manim import *

class KroneckerCompression(Scene):

    def construct(self):

        #################################################
        # TITLE
        #################################################

        title = Text(
            "Matrix Compression via Kronecker Product",
            font_size=42
        )

        self.play(Write(title))
        self.wait(2)
        self.play(FadeOut(title))


        #################################################
        # STEP 1 : ORIGINAL MATRIX
        #################################################

        step = Text("Step 1: Original Weight Matrix W", font_size=34).to_edge(UP)

        W = Matrix([
            [1,2,2,4],
            [3,4,6,8],
            [5,10,1,2],
            [15,20,3,4]
        ])

        label = MathTex("W").next_to(W, UP)

        self.play(Write(step))
        self.play(Create(W), Write(label))
        self.wait(2)


        #################################################
        # STEP 2 : PARTITION INTO BLOCKS
        #################################################

        new_step = Text("Step 2: Partition W into 2×2 blocks", font_size=34).to_edge(UP)
        self.play(Transform(step, new_step))

        e = W.get_entries()

        block1 = SurroundingRectangle(VGroup(e[0],e[1],e[4],e[5]),color=BLUE)
        block2 = SurroundingRectangle(VGroup(e[2],e[3],e[6],e[7]),color=GREEN)
        block3 = SurroundingRectangle(VGroup(e[8],e[9],e[12],e[13]),color=RED)
        block4 = SurroundingRectangle(VGroup(e[10],e[11],e[14],e[15]),color=YELLOW)

        self.play(Create(block1))
        self.play(Create(block2))
        self.play(Create(block3))
        self.play(Create(block4))
        self.wait(2)

        self.play(
            FadeOut(block1),
            FadeOut(block2),
            FadeOut(block3),
            FadeOut(block4),
            FadeOut(W),
            FadeOut(label)
        )


        #################################################
        # STEP 3 : SHOW SUBBLOCKS
        #################################################

        new_step = Text("Step 3: Extract Sub-blocks", font_size=34).to_edge(UP)
        self.play(Transform(step,new_step))

        W11 = Matrix([[1,2],[3,4]])
        W21 = Matrix([[5,10],[15,20]])
        W12 = Matrix([[2,4],[6,8]])
        W22 = Matrix([[1,2],[3,4]])

        blocks = VGroup(W11,W21,W12,W22).arrange(RIGHT,buff=1)

        self.play(Create(blocks))
        self.wait(2)

        self.play(FadeOut(blocks))


        #################################################
        # STEP 4 : VECTORIZE BLOCKS
        #################################################

        new_step = Text("Step 4: Vectorize Each Block", font_size=34).to_edge(UP)
        self.play(Transform(step,new_step))

        v1 = Matrix([[1,2,3,4]])
        v2 = Matrix([[5,10,15,20]])
        v3 = Matrix([[2,4,6,8]])
        v4 = Matrix([[1,2,3,4]])

        vectors = VGroup(v1,v2,v3,v4).arrange(DOWN,buff=0.7)

        self.play(Create(vectors))
        self.wait(2)

        self.play(FadeOut(vectors))


        #################################################
        # STEP 5 : REARRANGED MATRIX
        #################################################

        new_step = Text("Step 5: Build Rearranged Matrix W~", font_size=34).to_edge(UP)
        self.play(Transform(step,new_step))

        Wtilde = Matrix([
            [1,2,3,4],
            [5,10,15,20],
            [2,4,6,8],
            [1,2,3,4]
        ])

        label = MathTex(r"\tilde{W}").next_to(Wtilde,UP)

        self.play(Create(Wtilde),Write(label))
        self.wait(2)

        self.play(FadeOut(Wtilde),FadeOut(label))


        #################################################
        # STEP 6 : RANK 1 STRUCTURE
        #################################################

        new_step = Text("Step 6: Rank-1 Structure", font_size=34).to_edge(UP)
        self.play(Transform(step,new_step))

        text = Tex(
            "Rows are scaled versions of [1,2,3,4]",
            font_size=32
        )

        self.play(Write(text))
        self.wait(2)
        self.play(FadeOut(text))


        #################################################
        # STEP 7 : EXTRACT u AND v
        #################################################

        new_step = Text("Step 7: Extract vectors u and v", font_size=34).to_edge(UP)
        self.play(Transform(step,new_step))

        u = Matrix([[1],[5],[2],[1]])
        v = Matrix([[1],[2],[3],[4]])

        u_label = MathTex("u").next_to(u,UP)
        v_label = MathTex("v").next_to(v,UP)

        uv = VGroup(u,u_label,v,v_label).arrange(RIGHT,buff=2)

        self.play(Create(uv))
        self.wait(2)

        self.play(FadeOut(uv))


        #################################################
        # STEP 8 : RESHAPE INTO A AND B
        #################################################

        new_step = Text("Step 8: Reshape into A and B", font_size=34).to_edge(UP)
        self.play(Transform(step,new_step))

        A = Matrix([[1,2],[5,1]])
        B = Matrix([[1,2],[3,4]])

        A_label = MathTex("A").next_to(A,UP)
        B_label = MathTex("B").next_to(B,UP)

        AB = VGroup(A,A_label,B,B_label).arrange(RIGHT,buff=2)

        self.play(Create(AB))
        self.wait(2)

        self.play(FadeOut(AB))


        #################################################
        # STEP 9 : KRONECKER RECONSTRUCTION
        #################################################

        new_step = Text("Step 9: Reconstruct W = A ⊗ B", font_size=34).to_edge(UP)
        self.play(Transform(step,new_step))

        W_reconstructed = Matrix([
            [1,2,2,4],
            [3,4,6,8],
            [5,10,1,2],
            [15,20,3,4]
        ])

        label = MathTex("A \\otimes B").next_to(W_reconstructed,UP)

        self.play(Create(W_reconstructed),Write(label))
        self.wait(2)

        self.play(FadeOut(W_reconstructed),FadeOut(label))


        #################################################
        # FINAL MESSAGE
        #################################################

        final = Text(
            "Perfect Reconstruction — Compression Achieved!",
            font_size=36
        )

        self.play(Write(final))
        self.wait(3)