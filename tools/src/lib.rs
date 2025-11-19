#![allow(non_snake_case, unused_macros)]

use proconio::{input, marker::Chars};
use rand::prelude::*;
use std::{
    collections::{HashMap, VecDeque},
    ops::RangeBounds,
};
use svg::node::element::{Group, Line, Rectangle, Style, Symbol, Text, Title};

pub trait SetMinMax {
    fn setmin(&mut self, v: Self) -> bool;
    fn setmax(&mut self, v: Self) -> bool;
}
impl<T> SetMinMax for T
where
    T: PartialOrd,
{
    fn setmin(&mut self, v: T) -> bool {
        *self > v && {
            *self = v;
            true
        }
    }
    fn setmax(&mut self, v: T) -> bool {
        *self < v && {
            *self = v;
            true
        }
    }
}

#[macro_export]
macro_rules! mat {
	($($e:expr),*) => { Vec::from(vec![$($e),*]) };
	($($e:expr,)*) => { Vec::from(vec![$($e),*]) };
	($e:expr; $d:expr) => { Vec::from(vec![$e; $d]) };
	($e:expr; $d:expr $(; $ds:expr)+) => { Vec::from(vec![mat![$e $(; $ds)*]; $d]) };
}

#[derive(Clone, Debug)]
pub struct Input {
    pub N: usize,
    pub K: usize,
    pub T: usize,
    pub wall_v: Vec<Vec<bool>>,
    pub wall_h: Vec<Vec<bool>>,
    pub xy: Vec<(usize, usize)>,
}

impl std::fmt::Display for Input {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{} {} {}", self.N, self.K, self.T)?;
        for v in &self.wall_v {
            writeln!(
                f,
                "{}",
                v.iter()
                    .map(|&x| if x { '1' } else { '0' })
                    .collect::<String>()
            )?;
        }
        for h in &self.wall_h {
            writeln!(
                f,
                "{}",
                h.iter()
                    .map(|&x| if x { '1' } else { '0' })
                    .collect::<String>()
            )?;
        }
        for &(x, y) in &self.xy {
            writeln!(f, "{} {}", x, y)?;
        }
        Ok(())
    }
}

pub fn parse_input(f: &str) -> Input {
    let f = proconio::source::once::OnceSource::from(f);
    input! {
        from f,
        N: usize, K: usize, T: usize,
        wall_v: [Chars; N],
        wall_h: [Chars; N-1],
        xy: [(usize, usize); K],
    }
    let wall_v = wall_v
        .into_iter()
        .map(|s| s.into_iter().map(|c| c == '1').collect())
        .collect();
    let wall_h = wall_h
        .into_iter()
        .map(|s| s.into_iter().map(|c| c == '1').collect())
        .collect();
    Input {
        N,
        K,
        T,
        wall_v,
        wall_h,
        xy,
    }
}

pub fn read<T: Copy + PartialOrd + std::fmt::Display + std::str::FromStr, R: RangeBounds<T>>(
    token: Option<&str>,
    range: R,
) -> Result<T, String> {
    if let Some(v) = token {
        if let Ok(v) = v.parse::<T>() {
            if !range.contains(&v) {
                Err(format!("Out of range: {}", v))
            } else {
                Ok(v)
            }
        } else {
            Err(format!("Parse error: {}", v))
        }
    } else {
        Err("Unexpected EOF".to_owned())
    }
}

pub struct Output {
    pub C: usize,
    pub Q: usize,
    pub s: Vec<Vec<usize>>,
    pub rule: HashMap<(usize, usize), (usize, usize, usize)>,
}

pub fn parse_output(input: &Input, f: &str) -> Result<Output, String> {
    let mut f = f.split_whitespace();
    let C = read(f.next(), 1..=input.N * input.N * input.N * input.N)?;
    let Q = read(f.next(), 1..=input.N * input.N * input.N * input.N)?;
    let M = read(f.next(), 0..=input.T)?;
    let mut s = mat![0; input.N; input.N];
    for i in 0..input.N {
        for j in 0..input.N {
            s[i][j] = read(f.next(), 0..C)?;
        }
    }
    let mut rule = HashMap::new();
    for _ in 0..M {
        let c = read(f.next(), 0..C)?;
        let q = read(f.next(), 0..Q)?;
        let a = read(f.next(), 0..C)?;
        let s = read(f.next(), 0..Q)?;
        let d = read(f.next(), 'A'..='Z')?;
        let Some(d) = CMD.iter().position(|&x| x == d) else {
            return Err(format!("Invalid direction: {}", d));
        };
        if rule.contains_key(&(c, q)) {
            return Err(format!("Duplicate rule for ({}, {})", c, q));
        }
        rule.insert((c, q), (a, s, d));
    }
    if f.next().is_some() {
        return Err("The number of rules exceeded M".to_owned());
    }
    Ok(Output { C, Q, s, rule })
}

pub fn compute_X(
    N: usize,
    wall_v: &Vec<Vec<bool>>,
    wall_h: &Vec<Vec<bool>>,
    xy: &Vec<(usize, usize)>,
) -> i32 {
    let mut X = 0i32;
    for k in 0..xy.len() - 1 {
        let mut visited = mat![false; N; N];
        let mut que = VecDeque::new();
        que.push_back((xy[k], 0));
        visited[xy[k].0][xy[k].1] = true;
        loop {
            let ((i, j), d) = que.pop_front().unwrap();
            if (i, j) == xy[k + 1] {
                X += d;
                break;
            }
            for (di, dj) in DIJ {
                let (i2, j2) = (i + di, j + dj);
                if i2 >= N || j2 >= N || visited[i2][j2] {
                    continue;
                }
                if di == 0 {
                    if wall_v[i][j.min(j2)] {
                        continue;
                    }
                } else {
                    if wall_h[i.min(i2)][j] {
                        continue;
                    }
                }
                visited[i2][j2] = true;
                que.push_back(((i2, j2), d + 1));
            }
        }
    }
    X
}

pub fn gen(seed: u64) -> Input {
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(seed);
    let N = rng.gen_range(10i32..=20) as usize;
    let W = rng.gen_range(0..N as i32) as usize;
    let mut wall_v = mat![false; N; N - 1];
    let mut wall_h = mat![false; N - 1; N];
    let mut used = mat![false; N + 1; N + 1];
    for i in 0..=N {
        for j in 0..=N {
            if i == 0 || j == 0 || i == N || j == N {
                used[i][j] = true;
            }
        }
    }
    for _ in 0..W {
        let mut free = vec![];
        for i in 1..N {
            for j in 1..N {
                if !used[i][j] {
                    free.push((i, j));
                }
            }
        }
        let (i, j) = *free.choose(&mut rng).unwrap();
        let dir = rng.gen_range(0..4i32) as usize;
        let mut len = 0;
        let mut pi = i;
        let mut pj = j;
        loop {
            len += 1;
            pi += DIJ[dir].0;
            pj += DIJ[dir].1;
            if used[pi][pj] {
                break;
            }
        }
        match dir {
            0 => {
                for d in 0..len {
                    wall_v[i - d - 1][j - 1] = true;
                    used[i - d][j] = true;
                }
            }
            1 => {
                for d in 0..len {
                    wall_v[i + d][j - 1] = true;
                    used[i + d][j] = true;
                }
            }
            2 => {
                for d in 0..len {
                    wall_h[i - 1][j - d - 1] = true;
                    used[i][j - d] = true;
                }
            }
            3 => {
                for d in 0..len {
                    wall_h[i - 1][j + d] = true;
                    used[i][j + d] = true;
                }
            }
            _ => unreachable!(),
        }
    }
    let K = rng.gen_range(N as i32..=(N * N) as i32) as usize;
    let mut xy = vec![];
    for i in 0..N {
        for j in 0..N {
            xy.push((i, j));
        }
    }
    xy.shuffle(&mut rng);
    xy.truncate(K);
    let X = compute_X(N, &wall_v, &wall_h, &xy);
    let T = rng.gen_range(X as i32..=X as i32 * 2) as usize;
    Input {
        N,
        K,
        T,
        wall_v,
        wall_h,
        xy,
    }
}

pub const CMD: [char; 5] = ['U', 'D', 'L', 'R', 'S'];
pub const DIJ: [(usize, usize); 5] = [(!0, 0), (1, 0), (0, !0), (0, 1), (0, 0)];

fn can_move(input: &Input, (i, j): (usize, usize), dir: usize) -> bool {
    if dir == 4 {
        return true;
    }
    let (di, dj) = DIJ[dir];
    let i2 = i + di;
    let j2 = j + dj;
    if i2 >= input.N || j2 >= input.N {
        return false;
    }
    if di == 0 {
        !input.wall_v[i][j.min(j2)]
    } else {
        !input.wall_h[i.min(i2)][j]
    }
}

pub struct Sim {
    pub turn: usize,
    pub target: usize,
    pub p: (usize, usize),
    pub q: usize,
    pub cs: Vec<Vec<usize>>,
    pub route: Vec<(usize, usize)>,
    pub count: HashMap<(usize, usize), usize>,
    pub history: Vec<(usize, usize, usize, usize, usize)>,
}

impl Sim {
    pub fn new(input: &Input, out: &Output) -> Self {
        Sim {
            turn: 0,
            target: 1,
            p: input.xy[0],
            q: 0,
            cs: out.s.clone(),
            route: vec![input.xy[0]],
            count: HashMap::new(),
            history: vec![],
        }
    }
    pub fn step(&mut self, input: &Input, out: &Output) -> bool {
        let c = self.cs[self.p.0][self.p.1];
        let Some(&(a, s, d)) = out.rule.get(&(c, self.q)) else {
            return true;
        };
        *self.count.entry((c, self.q)).or_insert(0) += 1;
        self.history.push((c, self.q, a, s, d));
        self.turn += 1;
        self.cs[self.p.0][self.p.1] = a;
        self.q = s;
        if can_move(input, self.p, d) {
            let (di, dj) = DIJ[d];
            self.p.0 += di;
            self.p.1 += dj;
            self.route.push(self.p);
            if self.p == input.xy[self.target] {
                self.target += 1;
                self.route.clear();
                self.route.push(self.p);
                if self.target == input.xy.len() {
                    return true;
                }
            }
        }
        false
    }
}

pub fn compute_score(input: &Input, out: &Output) -> (i64, String) {
    let (mut score, err, _) = compute_score_details(input, &out, input.T);
    if err.len() > 0 {
        score = 0;
    }
    (score, err)
}

pub fn compute_score_details(input: &Input, out: &Output, t: usize) -> (i64, String, Sim) {
    let mut sim = Sim::new(input, out);
    for _ in 0..t {
        if sim.step(input, out) {
            break;
        }
    }
    let score = if sim.target == input.K {
        out.C + out.Q
    } else {
        2 * input.N.pow(4) + (input.K - sim.target) * input.N.pow(2)
    };
    (score as i64, String::new(), sim)
}

pub fn text_color_for_background(r: f64, g: f64, b: f64) -> &'static str {
    let luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    if luminance > 150.0 {
        "#000000"
    } else {
        "#ffffff"
    }
}

/// 0 <= val <= 1
pub fn color(mut val: f64) -> (String, &'static str) {
    val.setmin(1.0);
    val.setmax(0.0);
    let (r, g, b) = if val < 0.5 {
        let x = val * 2.0;
        (
            30. * (1.0 - x) + 144. * x,
            144. * (1.0 - x) + 255. * x,
            255. * (1.0 - x) + 30. * x,
        )
    } else {
        let x = val * 2.0 - 1.0;
        (
            144. * (1.0 - x) + 255. * x,
            255. * (1.0 - x) + 30. * x,
            30. * (1.0 - x) + 70. * x,
        )
    };
    (
        format!(
            "#{:02x}{:02x}{:02x}",
            r.round() as i32,
            g.round() as i32,
            b.round() as i32
        ),
        text_color_for_background(r, g, b),
    )
}

pub fn rect(x: usize, y: usize, w: usize, h: usize, fill: &str) -> Rectangle {
    Rectangle::new()
        .set("x", x)
        .set("y", y)
        .set("width", w)
        .set("height", h)
        .set("fill", fill)
}

pub fn group(title: String) -> Group {
    Group::new().add(Title::new(title))
}

pub fn vis_default(input: &Input, out: &Output) -> (i64, String, String) {
    let (mut score, err, svg, _) = vis(input, &out, input.T, false, false);
    if err.len() > 0 {
        score = 0;
    }
    (score, err, svg)
}

// https://www.svgrepo.com/svg/368071/flag
const FLAG: &str = r#"<path d="M18.383 4.318c-.374-.155-.804-.069-1.09.217-1.264 1.263-3.321 1.264-4.586 0-2.045-2.043-5.37-2.043-7.414 0-.188.187-.293.442-.293.707v13c0 .552.447 1 1 1s1-.448 1-1v-4.553c1.271-.997 3.121-.911 4.293.26 2.045 2.043 5.371 2.043 7.414 0 .188-.188.293-.442.293-.707v-8c0-.405-.244-.769-.617-.924z"/>"#;

// https://www.svgrepo.com/svg/275995/robot
const ROBOT: &str = r#"<rect x="245.33" y="15.938" style="fill:#CCD1D9;" width="21.344" height="84.19"/>
<g>
	<path style="fill:#434A54;" d="M74.672,309.344V416c0,5.891,4.766,10.672,10.656,10.672S96,421.891,96,416V309.344H74.672z"/>
	<path style="fill:#434A54;" d="M416,309.344v106.328c0,5.891,4.781,10.672,10.672,10.672s10.656-4.781,10.656-10.672V309.344H416z"
		/>
</g>
<g>
	<path style="fill:#CCD1D9;" d="M85.328,234.672h71.109V320H85.328c-23.562,0-42.656-19.094-42.656-42.656
		C42.672,253.766,61.766,234.672,85.328,234.672z"/>
	<path style="fill:#CCD1D9;" d="M426.672,234.672h-71.109V320h71.109c23.562,0,42.656-19.094,42.656-42.656
		C469.328,253.766,450.234,234.672,426.672,234.672z"/>
</g>
<circle style="fill:#434A54;" cx="256" cy="426.67" r="85.33"/>
<path style="fill:#E6E9ED;" d="M405.328,245.344C405.328,162.859,338.469,96,256,96s-149.328,66.859-149.328,149.344V448
	c0,11.781,9.547,21.344,21.328,21.344h256c11.781,0,21.328-9.562,21.328-21.344V245.344L405.328,245.344z"/>
<path style="fill:#AAB2BD;" d="M106.672,245.344V256h298.656v-10.656l0,0c0-3.594-0.125-7.156-0.375-10.672H107.047
	C106.797,238.188,106.672,241.75,106.672,245.344z"/>
<circle style="fill:#E6E9ED;" cx="256" cy="32" r="32"/>
<path style="fill:#A0D468;" d="M320,309.344c0,5.891-4.781,10.656-10.672,10.656s-10.656-4.766-10.656-10.656
	s4.766-10.672,10.656-10.672S320,303.453,320,309.344z"/>
<path style="fill:#434A54;" d="M218.453,181.344c0,5.891-4.781,10.656-10.672,10.656s-10.672-4.766-10.672-10.656
	s4.781-10.672,10.672-10.672S218.453,175.453,218.453,181.344z"/>
<path style="fill:#ED5565;" d="M362.672,309.344c0,5.891-4.781,10.656-10.672,10.656s-10.672-4.766-10.672-10.656
	s4.781-10.672,10.672-10.672S362.672,303.453,362.672,309.344z"/>
<g>
	<path style="fill:#434A54;" d="M303.781,181.344c0,5.891-4.781,10.656-10.672,10.656s-10.656-4.766-10.656-10.656
		s4.766-10.672,10.656-10.672S303.781,175.453,303.781,181.344z"/>
	<path style="fill:#434A54;" d="M202.672,298.672H160c-5.891,0-10.672,4.781-10.672,10.672S154.109,320,160,320h42.672
		c5.891,0,10.656-4.766,10.656-10.656S208.562,298.672,202.672,298.672z"/>
</g>"#;

pub fn vis(
    input: &Input,
    out: &Output,
    t: usize,
    show_number: bool,
    show_targets: bool,
) -> (i64, String, String, Sim) {
    let D = 600 / input.N;
    let W = input.N * D;
    let H = input.N * D;
    let (score, err, sim) = compute_score_details(input, out, t);
    let mut doc = svg::Document::new()
        .set("id", "vis")
        .set("viewBox", (-5, -5, W + 10, H + 10))
        .set("width", W + 10)
        .set("height", H + 10)
        .set("style", "background-color:white");
    doc = doc.add(Style::new(format!(
        "text {{text-anchor: middle;dominant-baseline: central;}}"
    )));
    doc = doc.add(
        Symbol::new()
            .set("id", "flag")
            .set("viewBox", (0, 0, 24.0, 24.0))
            .add(svg::node::Blob::new(FLAG)),
    );
    doc = doc.add(
        Symbol::new()
            .set("id", "robot")
            .set("viewBox", (0, 0, 512, 512))
            .add(svg::node::Blob::new(ROBOT)),
    );
    let font_size = D / (out.C - 1).to_string().len().max(2);
    for i in 0..input.N {
        for j in 0..input.N {
            let mut g = group(format!("({}, {})\nc = {}", i, j, sim.cs[i][j]));
            let (bg, fg) = color(sim.cs[i][j] as f64 / out.C as f64);
            g = g.add(rect(j * D, i * D, D, D, &bg));
            if show_number && !show_targets {
                g = g.add(
                    Text::new(format!("{}", sim.cs[i][j]))
                        .set("x", j * D + D / 2)
                        .set("y", i * D + D / 2)
                        .set("font-size", font_size)
                        .set("fill", fg),
                );
            }
            doc = doc.add(g);
        }
    }
    if show_targets {
        for k in sim.target..input.K {
            let p = input.xy[k];
            let color = if k == sim.target {
                "#000000a0"
            } else {
                "#00000050"
            };
            doc = doc.add(
                svg::node::element::Use::new()
                    .set("href", "#flag")
                    .set("x", p.1 * D)
                    .set("y", p.0 * D)
                    .set("width", D)
                    .set("height", D)
                    .set("fill", color),
            );
            if show_number {
                doc = doc.add(
                    Text::new(format!("{}", k))
                        .set("x", p.1 * D + D / 2)
                        .set("y", p.0 * D + D / 3)
                        .set("font-size", D / 3)
                        .set("fill", "white"),
                );
            }
        }
    }
    for i in 0..=input.N {
        doc = doc.add(
            Line::new()
                .set("x1", 0)
                .set("y1", i * D)
                .set("x2", input.N * D)
                .set("y2", i * D)
                .set(
                    "stroke",
                    if i == 0 || i == input.N {
                        "black"
                    } else {
                        "lightgray"
                    },
                )
                .set("stroke-width", if i == 0 || i == input.N { 2 } else { 1 }),
        );
        doc = doc.add(
            Line::new()
                .set("x1", i * D)
                .set("y1", 0)
                .set("x2", i * D)
                .set("y2", input.N * D)
                .set(
                    "stroke",
                    if i == 0 || i == input.N {
                        "black"
                    } else {
                        "lightgray"
                    },
                )
                .set("stroke-width", if i == 0 || i == input.N { 2 } else { 1 }),
        );
    }
    for i in 0..input.N {
        for j in 0..input.N {
            if j + 1 < input.N {
                if input.wall_v[i][j] {
                    doc = doc.add(
                        Line::new()
                            .set("x1", j * D + D)
                            .set("y1", i * D)
                            .set("x2", j * D + D)
                            .set("y2", (i + 1) * D)
                            .set("stroke", "black")
                            .set("stroke-width", 2),
                    );
                }
            }
            if i + 1 < input.N {
                if input.wall_h[i][j] {
                    doc = doc.add(
                        Line::new()
                            .set("x1", j * D)
                            .set("y1", i * D + D)
                            .set("x2", (j + 1) * D)
                            .set("y2", i * D + D)
                            .set("stroke", "black")
                            .set("stroke-width", 2),
                    );
                }
            }
        }
    }
    if sim.route.len() > 1 {
        for k in 0..sim.route.len() - 1 {
            let p1 = sim.route[k];
            let p2 = sim.route[k + 1];
            doc = doc.add(
                Line::new()
                    .set("x1", p1.1 * D + D / 2)
                    .set("y1", p1.0 * D + D / 2)
                    .set("x2", p2.1 * D + D / 2)
                    .set("y2", p2.0 * D + D / 2)
                    .set("stroke", "#00000080")
                    .set("stroke-width", D / 10),
            );
        }
    }
    if sim.target < input.xy.len() && !show_targets {
        let p = input.xy[sim.target];
        doc = doc.add(
            svg::node::element::Use::new()
                .set("href", "#flag")
                .set("x", p.1 * D)
                .set("y", p.0 * D)
                .set("width", D)
                .set("height", D)
                .set("fill", "#000000a0"),
        );
    }
    doc = doc.add(
        svg::node::element::Use::new()
            .set("href", "#robot")
            .set("x", sim.p.1 * D + D / 8)
            .set("y", sim.p.0 * D + D / 8)
            .set("width", D * 3 / 4)
            .set("height", D * 3 / 4),
    );
    (score, err, doc.to_string(), sim)
}
