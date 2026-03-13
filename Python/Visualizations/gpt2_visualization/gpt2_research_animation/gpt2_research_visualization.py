from manim import *

class TransformerTower(ThreeDScene):

    def construct(self):

        self.set_camera_orientation(phi=65*DEGREES, theta=-40*DEGREES)

        title = Text("GPT-2 Small Transformer", font_size=50)
        title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)

        layers = VGroup()

        for i in range(12):

            layer = Prism(
                dimensions=[5,0.4,2],
                fill_color=GRAY,
                fill_opacity=0.25
            )

            layer.shift(DOWN*i*0.7)

            layers.add(layer)

        self.play(LaggedStart(*[Create(l) for l in layers], lag_ratio=0.1))

        label = Text("12 Transformer Layers", font_size=30)
        label.to_edge(RIGHT)
        self.add_fixed_in_frame_mobjects(label)

        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait(5)
class TokenFlow(ThreeDScene):

    def construct(self):

        self.set_camera_orientation(phi=60*DEGREES, theta=-30*DEGREES)

        tokens = VGroup()

        words = ["To","date","the","cleverest","thinker"]

        for i,w in enumerate(words):

            t = Square(0.4,color=BLUE)
            label = Text(w,font_size=18)

            group = VGroup(t,label).arrange(DOWN)

            group.shift(LEFT*4 + RIGHT*i)

            tokens.add(group)

        self.play(Create(tokens))

        layers = VGroup()

        for i in range(6):

            layer = Rectangle(width=6,height=0.3,color=GRAY)

            layer.shift(DOWN*i*0.8)

            layers.add(layer)

        self.play(Create(layers))

        for token in tokens:

            self.play(token.animate.shift(DOWN*4), run_time=2)

        self.wait()

class AttentionHeads(Scene):

    def construct(self):

        title = Text("Multi-Head Attention", font_size=50)
        title.to_edge(UP)

        self.play(Write(title))

        input_nodes = VGroup()

        for i in range(6):
            node = Dot(LEFT*3 + UP*(i-3))
            input_nodes.add(node)

        output_nodes = VGroup()

        for i in range(6):
            node = Dot(RIGHT*3 + UP*(i-3))
            output_nodes.add(node)

        self.play(Create(input_nodes),Create(output_nodes))

        lines = VGroup()

        for a in input_nodes:
            for b in output_nodes:
                line = Line(a,b,stroke_width=1)
                lines.add(line)

        self.play(Create(lines), run_time=3)

        self.wait()


class MatrixHeatmap(Scene):

    def construct(self):

        title = Text("Weight Matrix (MLP)", font_size=50)
        title.to_edge(UP)

        self.play(Write(title))

        grid = VGroup()

        for i in range(15):
            for j in range(15):

                val = np.random.rand()

                color = interpolate_color(BLUE,RED,val)

                square = Square(
                    0.3,
                    fill_color=color,
                    fill_opacity=1,
                    stroke_width=0
                )

                square.move_to([j*0.35-2.5,i*0.35-2.5,0])

                grid.add(square)

        self.play(FadeIn(grid))

        self.wait(2)

class MLPCompression(Scene):

    def construct(self):

        title = Text("MLP Compression", font_size=50)
        title.to_edge(UP)

        self.play(Write(title))

        matrix = Rectangle(width=4,height=3,color=BLUE)

        label = Text("W (768 x 3072)", font_size=30)

        group = VGroup(matrix,label).arrange(DOWN)

        self.play(Create(group))

        svd = MathTex("W = U \\Sigma V^T", font_size=60)

        self.play(Transform(group,svd))

        self.wait(2)

class KroneckerCompression(Scene):

    def construct(self):

        title = Text("Kronecker Factorization", font_size=50)
        title.to_edge(UP)

        self.play(Write(title))

        A = Matrix([["a11","a12"],["a21","a22"]])
        B = Matrix([["b11","b12"],["b21","b22"]])

        kron = MathTex("A \\otimes B")

        group = VGroup(A,kron,B).arrange(RIGHT,buff=1)

        self.play(Write(group))

        self.wait(3)

class CompressionComparison(Scene):

    def construct(self):

        title = Text("Model Size Reduction", font_size=50)
        title.to_edge(UP)

        self.play(Write(title))

        before = Rectangle(width=3,height=5,color=RED)
        after = Rectangle(width=2,height=3,color=GREEN)

        before_text = Text("124M Params",font_size=28)
        after_text = Text("40M Params",font_size=28)

        b = VGroup(before,before_text).arrange(DOWN)
        a = VGroup(after,after_text).arrange(DOWN)

        group = VGroup(b,a).arrange(RIGHT,buff=3)

        self.play(Create(group))

        arrow = Arrow(b.get_right(),a.get_left())

        self.play(GrowArrow(arrow))

        self.wait()