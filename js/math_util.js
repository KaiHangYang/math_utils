/*** Part of the functions are copied and modified from https://github.com/sloisel/numeric/blob/master/src/numeric.js ***/
_m = {};

(function() {
    function matrix_to32fv(x) {
        var m = x.length, n = x[0].length;
        if (typeof n == "undefined") {
            return new Float32Array(x);
        }
    
        var result = new Float32Array(m*n);
        var cnt = 0;
        for (var i = 0; i < m; ++i) {
            for (var j = 0; j < n; ++j) {
                result[cnt] = x[i][j];
                cnt += 1;
            }
        }
        return result;
    }
    
    function matrix_from32fv(x, size) {
        var m, n;
        if (size instanceof Array || size instanceof Float32Array) {
            m = size[0];
            n = size[1];
        }
        else {
            m = size;
            n = size;
        }
        var mat = new Array(m);
        var cnt = 0;
        for (var i = 0; i < m; ++i) {
            mat[i] = new Array(n);
            for (var j = 0; j < n; ++j) {
                mat[i][j] = x[cnt];
                cnt += 1;
            }
        }
        return mat;
    }
    
    function matrix_transpose(x) {
        var i,j,m = x.length,n = x[0].length, ret=Array(n),A0,A1,Bj;
        for(j=0;j<n;j++) ret[j] = Array(m);
        for(i=m-1;i>=1;i-=2) {
            A1 = x[i];
            A0 = x[i-1];
            for(j=n-1;j>=1;--j) {
                Bj = ret[j]; Bj[i] = A1[j]; Bj[i-1] = A0[j];
                --j;
                Bj = ret[j]; Bj[i] = A1[j]; Bj[i-1] = A0[j];
            }
            if(j===0) {
                Bj = ret[0]; Bj[i] = A1[0]; Bj[i-1] = A0[0];
            }
        }
        if(i===0) {
            A0 = x[0];
            for(j=n-1;j>=1;--j) {
                ret[j][0] = A0[j];
                --j;
                ret[j][0] = A0[j];
            }
            if(j===0) { ret[0][0] = A0[0]; }
        }
        return ret;
    }
    
    function matrix_fill(s, v) {
        var m, n;
        if (s instanceof Array || s instanceof Float32Array) {
            m = s[0];
            n = s[1];
        }
        else {
            m = s;
            n = s;
        }
    
        var matrix = new Array(m);
        for (var i = 0; i < m; ++i) {
            matrix[i] = new Array(n);
            for (var j = 0; j < n; ++j) {
                matrix[i][j] = v;
            }
        }
        return matrix;
    }
    
    function matrix_identity(s) {
        var identity = new Array(s);
        for (var i = 0; i < s; ++i) {
            identity[i] = new Array(s);
            for (var j = 0; j < s; ++j) {
                identity[i][j] = 0.0;
            }
            identity[i][i] = 1.0
        }
    
        return identity;
    }
    
    function matrix_clone(mat) {
        var m = mat.length, n = mat[0].length;
        var c_mat = Array(m);
        for (var i = 0; i < m; ++i) {
            c_mat[i] = new Array(n);
            for (var j = 0; j < n; ++j) {
                c_mat[i][j] = mat[i][j]
            }
        }
    
        return c_mat;
    }
    
    function normfv(v) {
        var norm = vector_norm(v);
        return Array.prototype.map.call(v, function(a) {return a / norm;});
    }
    
    function deg2rad(deg) {
        return deg * Math.PI / 180;
    }
    
    function vector_minus(vec_a, vec_b) {
        vec_a = Array.prototype.slice.call(vec_a);
        return vec_a.map(function(x, idx) {return x - vec_b[idx]});
    }
    function vector_add(vec_a, vec_b) {
        vec_a = Array.prototype.slice.call(vec_a);
        return vec_a.map(function(x, idx) {return x + vec_b[idx]});
    }
    function vector_frac(vec_a, v) {
        return Array.prototype.map.call(vec_a, function(a) {return a / v});
    }
    function vector_norm(v) {
        var norm = Math.sqrt(Array.prototype.map.call(v, function(a) {return a * a;}).reduce(function(a, b) {return a + b;}));
        return norm;
    }
    
    // Only support vec3 cross
    function vector_cross(vec_a, vec_b) {
        var vec_c = new Float32Array(vec_a.length);
        vec_c[0] = vec_a[1] * vec_b[2] - vec_a[2] * vec_b[1];
        vec_c[1] = -(vec_a[0] * vec_b[2] - vec_a[2] * vec_b[0]);
        vec_c[2] = vec_a[0] * vec_b[1] - vec_a[1] * vec_b[0];
        return vec_c;
    }
    function vector_dot(vec_a, vec_b) {
        vec_a = Array.prototype.slice.call(vec_a);
        return vec_a.map(function(a, idx) {return a * vec_b[idx];}).reduce(function(a, b) {return a + b;});
    }
    
    function matrix_inverse(x) {
        var abs = Math.abs, m = x.length, n = x[0].length;
        var A = matrix_clone(x), Ai, Aj;
        var I = matrix_identity(m), Ii, Ij;
        var i,j,k,x;
        for(j=0;j<n;++j) {
            var i0 = -1;
            var v0 = -1;
            for(i=j;i!==m;++i) { k = abs(A[i][j]); if(k>v0) { i0 = i; v0 = k; } }
            Aj = A[i0]; A[i0] = A[j]; A[j] = Aj;
            Ij = I[i0]; I[i0] = I[j]; I[j] = Ij;
            x = Aj[j];
            for(k=j;k!==n;++k)    Aj[k] /= x; 
            for(k=n-1;k!==-1;--k) Ij[k] /= x;
            for(i=m-1;i!==-1;--i) {
                if(i!==j) {
                    Ai = A[i];
                    Ii = I[i];
                    x = Ai[j];
                    for(k=j+1;k!==n;++k)  Ai[k] -= Aj[k]*x;
                    for(k=n-1;k>0;--k) { Ii[k] -= Ij[k]*x; --k; Ii[k] -= Ij[k]*x; }
                    if(k===0) Ii[0] -= Ij[0]*x;
                }
            }
        }
        return I;
    }
    
    
    function perspective(fov, ratio, near, far) {
        var matrix = matrix_fill(4, 0);
        matrix[1][1] = 1.0 / Math.tan(fov/2.0);
        matrix[0][0] = matrix[1][1] / ratio;
        matrix[3][2] = -1;
        matrix[2][2] = -(far + near) / (far - near);
        matrix[2][3] = -2 * far * near / (far - near);
        return matrix;
    }
    
    function lookAt(eye, target, up) {
        eye = Array.prototype.slice.call(new Float32Array(eye));
        target = new Float32Array(target);
        up = new Float32Array(up);
    
        var Z = normfv(vector_minus(eye, target));
        var X = normfv(vector_cross(up, Z))
        var Y = vector_cross(Z, X)
        var mat = matrix_from32fv(new Float32Array([X[0],X[1],X[2],vector_dot(X, eye.map(function(a){return -a;})), Y[0],Y[1],Y[2],vector_dot(Y, eye.map(function(a){return -a;})), Z[0],Z[1],Z[2],vector_dot(Z, eye.map(function(a){return -a;})), 0., 0., 0., 1.]), 4);
    
        return mat
    }
    
    function translate(target) {
        return matrix_from32fv(new Float32Array([1, 0, 0, target[0], 0, 1, 0, target[1], 0, 0, 1, target[2], 0, 0, 0, 1]), 4);
    }
    
    function rotate(axis, theta) {
        var axis = normfv(axis);
        var u = axis[0];
        var v = axis[1];
        var w = axis[2];
        
        var mat = matrix_fill(4, 0.0);
        
        mat[0][0] = Math.cos(theta) + (u * u) * (1 - Math.cos(theta));
        mat[1][0] = u * v * (1 - Math.cos(theta)) + w * Math.sin(theta);
        mat[2][0] = u * w * (1 - Math.cos(theta)) - v * Math.sin(theta);
        mat[3][0] = 0;
        
        mat[0][1] = u * v * (1 - Math.cos(theta)) - w * Math.sin(theta);
        mat[1][1] = Math.cos(theta) + v * v * (1 - Math.cos(theta));
        mat[2][1] = w * v * (1 - Math.cos(theta)) + u * Math.sin(theta);
        mat[3][1] = 0;
        
        mat[0][2] = u * w * (1 - Math.cos(theta)) + v * Math.sin(theta);
        mat[1][2] = v * w * (1 - Math.cos(theta)) - u * Math.sin(theta);
        mat[2][2] = Math.cos(theta) + w * w * (1 - Math.cos(theta));
        mat[3][2] = 0;
        
        mat[0][3] = 0;
        mat[1][3] = 0;
        mat[2][3] = 0;
        mat[3][3] = 1;
    
        return mat;
    }
    
    function scale(ratio) {
        return matrix_from32fv(new Float32Array([ratio[0],0,0,0, 0,ratio[1],0,0, 0,0,ratio[2],0, 0,0,0,1]), 4);
    }
    
    function matrix_vector_mul(mat, vec) {
        var m = mat.length, n = mat[0].length;
        var result = new Float32Array(m);
    
        for (var i = 0; i < m; ++i) {
            result[i] = vector_dot(mat[i], vec);
        }
        return result;
    }
    
    function matrix_matrix_mul(mat_a, mat_b) {
        var m = mat_a.length, n = mat_b[0].length;
        var z = mat_a[0].length;
        var result = matrix_fill([m, n], 0.0);
    
        for (var i = 0; i < m; ++i) {
            for (var j = 0; j < n; ++j) {
                var tmp_ = [];
                for (var k = 0; k < z; ++k) {
                    tmp_[k] = mat_b[k][j];
                }
                result[i][j] = vector_dot(mat_a[i], tmp_);
            }
        }
        return result;
    }

    _m.matrix_to32fv = matrix_to32fv;
    _m.matrix_from32fv = matrix_from32fv;
    _m.matrix_transpose = matrix_transpose;
    _m.matrix_fill = matrix_fill;
    _m.matrix_identity = matrix_identity;
    _m.matrix_clone = matrix_clone;
    _m.normfv = normfv;
    _m.deg2rad = deg2rad;
    _m.vector_minus = vector_minus;
    _m.vector_add = vector_add;
    _m.vector_cross = vector_cross;
    _m.vector_dot = vector_dot;
    _m.vector_frac = vector_frac;
    _m.vector_norm = vector_norm;
    _m.matrix_inverse = matrix_inverse;
    _m.perspective = perspective;
    _m.lookAt = lookAt;
    _m.translate = translate;
    _m.rotate = rotate;
    _m.scale = scale;
    _m.matrix_vector_mul = matrix_vector_mul;
    _m.matrix_matrix_mul = matrix_matrix_mul;
})();

/**** Unit Test ****/
(function() {
    const assert = require("assert");

    function checknear(a, b) {
        assert.ok(a.length == b.length, "Array size doesn't match! a:" + a.length + " b:" + b.length);
        for (var i = 0; i < a.length; ++i) {
            if (Math.abs(a[i] - b[i]) > 0.0001) {
                console.log(a[i], b[i]);
                return false;
            }
        }
        return true;
    }

    var result;
    /******** Check matrix_inverse *******/
    result = _m.matrix_inverse(_m.matrix_from32fv(new Float32Array([0.11555905577932213, 0.738184501960425, 0.5804137287091379, 0.4699458894558518, 0.8095798004366057, 0.0829459627383573, 0.10706353602349172, 0.08305823306043891, 0.4650258290362618, 0.6944323567403545, 0.06483547976151582, 0.014406512798651794, 0.9751298776384586, 0.4517797277610748, 0.41371687160641346, 0.40297826091354716]), [4, 4]));
    assert.ok(checknear(_m.matrix_to32fv(result), [-0.29389774620637715, 1.0736758580152652, 0.10766733253408481, 0.11759271379069397, -0.4052469760188802, -1.7925825815884169, 1.573695050481297, 0.7858027721702396, 8.018690995387887, 15.066770412605207, -2.272401027873725, -12.375438850473472, -7.066874785593195, -16.056692594278125, 0.30814935207818495, 14.021227601882144]));

    result = _m.matrix_inverse(_m.matrix_from32fv(new Float32Array([0.11555905577932213, 0.738184501960425, 0.5804137287091379, 0.4699458894558518, 0.8095798004366057, 0.0829459627383573, 0.10706353602349172, 0.08305823306043891, 0.4650258290362618, 0.6944323567403545, 0.06483547976151582, 0.014406512798651794, 0.9751298776384586, 0.4517797277610748, 0.41371687160641346, 0.40297826091354716]), 4));
    assert.ok(checknear(_m.matrix_to32fv(result), [-0.29389774620637715, 1.0736758580152652, 0.10766733253408481, 0.11759271379069397, -0.4052469760188802, -1.7925825815884169, 1.573695050481297, 0.7858027721702396, 8.018690995387887, 15.066770412605207, -2.272401027873725, -12.375438850473472, -7.066874785593195, -16.056692594278125, 0.30814935207818495, 14.021227601882144]));
    console.log("matrix_inverse passed...");
    /*************************************/

    /********** Check perspective *********/
    result = _m.perspective(_m.deg2rad(32.78), 7/3.0, 0.001, 100000);
    assert.ok(checknear(_m.matrix_to32fv(result), [1.457100, 0.000000, 0.000000, 0.000000, 0.000000, 3.399899, 0.000000, 0.000000, 0.000000, 0.000000, -1.000000, -0.002000, 0.000000, 0.000000, -1.000000, 0.000000]));
    console.log("perspective passed...");
    /**************************************/

    /********** Check lookAt *********/
    result = _m.lookAt([1,3, 22], [7, 0, -10], [1, 0, 0]);
    assert.ok(checknear(_m.matrix_to32fv(result), [0.000000, -0.995634, 0.093341, 0.933407, 0.983018, 0.017129, 0.182710, -5.054024, -0.183511, 0.091756, 0.978726, -21.623728, 0.000000, 0.000000, 0.000000, 1.000000]));
    console.log("lookAt passed...");
    /*********************************/

    /********** Check translate *********/
    result = _m.translate([1, -73, 33]);
    assert.ok(checknear(_m.matrix_to32fv(result), [1.000000, 0.000000, 0.000000, 1.000000, 0.000000, 1.000000, 0.000000, -73.000000, 0.000000, 0.000000, 1.000000, 33.000000, 0.000000, 0.000000, 0.000000, 1.000000]));
    console.log("translate passed...");
    /*********************************/
    
    /********** Check rotate *********/
    result = _m.rotate([1, 33, 14], _m.deg2rad(33.78));
    assert.ok(checknear(_m.matrix_to32fv(result), [0.831310, -0.212731, 0.513487, 0.000000, 0.221396, 0.974139, 0.045145, 0.000000, -0.509812, 0.076154, 0.856909, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000]));
    console.log("rotate passed...");
    /*********************************/

    /********** Check scale *********/
    result = _m.scale([0.22, 33, 11]);
    assert.ok(checknear(_m.matrix_to32fv(result), [0.220000, 0.000000, 0.000000, 0.000000, 0.000000, 33.000000, 0.000000, 0.000000, 0.000000, 0.000000, 11.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000]));
    console.log("scale passed...");
    /*********************************/

    /********** Check scale *********/
    result = _m.matrix_vector_mul(_m.matrix_from32fv(new Float32Array([0.6435891901701479, 0.1579527774331445, 0.17350481126163197, 0.008636752596140829, 0.22169139897822787, 0.11569286461291428, 0.8636738491785446, 0.17937016639672898, 0.7299834869592117, 0.14348274367822833, 0.04545696274405886, 0.46068299528773393, 0.8488570140146383, 0.47268859794126095, 0.771141437599844, 0.8561945478198865, 0.8780069138717316, 0.37689846825077866, 0.2637592468874401, 0.4212065801293786, 0.12988857551913702, 0.8293071230552266, 0.9067303206938253, 0.3560755480545521, 0.35715181711114385, 0.5924683413601126, 0.842486508782473, 0.11767187866755846, 0.6655294401879299, 0.4041191603167539, 0.8670856527536362, 0.554235612762505, 0.6060844006809943, 0.7657074300026616, 0.9938555662278019, 0.20205511557688471, 0.16836479846300967, 0.07633536179889577, 0.9071177143605743, 0.11539501291841048, 0.5722914808251663, 0.3036484985806166, 0.8912098334131017, 0.09106311219794994, 0.6215155373202922, 0.3551420101165428, 0.9597023941138576, 0.9852556173932301, 0.11369306414214708, 0.3160880285273532, 0.25767636001700156, 0.14876763714492292, 0.7326381314414568, 0.05287543533819217, 0.20343746748118385, 0.5566745012654155, 0.134067977324708, 0.47108160361954365, 0.4134304293776355, 0.43549077743020015, 0.8628948885432544, 0.2952053536988597, 0.6143510820872318, 0.7700720940072747, 0.3553006512415755, 0.6917536180341274, 0.5514945432143873, 0.00812546525351876, 0.23428465611364568, 0.7596568881769677]), [10, 7]), new Float32Array([0.68036086, 0.09922524, 0.54862266, 0.49598948, 0.63678755, 0.03850587, 0.71774647]));
    assert.ok(checknear(_m.matrix_to32fv(result), [1.31854224, 0.96104703, 1.55564732, 1.52086901, 2.37226442, 0.95943791, 1.86317171, 1.12667514, 1.58257011, 1.77166548]));
    console.log("matrix_vector_mul passed...");
    /*********************************/
    result = _m.matrix_matrix_mul(_m.matrix_from32fv(new Float32Array([0.6435891901701479, 0.1579527774331445, 0.17350481126163197, 0.008636752596140829, 0.22169139897822787, 0.11569286461291428, 0.8636738491785446, 0.17937016639672898, 0.7299834869592117, 0.14348274367822833, 0.04545696274405886, 0.46068299528773393, 0.8488570140146383, 0.47268859794126095, 0.771141437599844, 0.8561945478198865, 0.8780069138717316, 0.37689846825077866, 0.2637592468874401, 0.4212065801293786, 0.12988857551913702, 0.8293071230552266, 0.9067303206938253, 0.3560755480545521, 0.35715181711114385, 0.5924683413601126, 0.842486508782473, 0.11767187866755846, 0.6655294401879299, 0.4041191603167539, 0.8670856527536362, 0.554235612762505, 0.6060844006809943, 0.7657074300026616, 0.9938555662278019, 0.20205511557688471, 0.16836479846300967, 0.07633536179889577, 0.9071177143605743, 0.11539501291841048, 0.5722914808251663, 0.3036484985806166, 0.8912098334131017, 0.09106311219794994, 0.6215155373202922, 0.3551420101165428, 0.9597023941138576, 0.9852556173932301, 0.11369306414214708, 0.3160880285273532, 0.25767636001700156, 0.14876763714492292, 0.7326381314414568, 0.05287543533819217, 0.20343746748118385, 0.5566745012654155, 0.134067977324708, 0.47108160361954365, 0.4134304293776355, 0.43549077743020015, 0.8628948885432544, 0.2952053536988597, 0.6143510820872318, 0.7700720940072747, 0.3553006512415755, 0.6917536180341274, 0.5514945432143873, 0.00812546525351876, 0.23428465611364568, 0.7596568881769677]), [10, 7]), _m.matrix_from32fv(new Float32Array([0.5547031691355524, 0.040976646528517624, 0.743338086951488, 0.941571448291056, 0.7963686688109199, 0.17798962598215073, 0.9570413379779039, 0.7222603472101974, 0.9627457411487461, 0.38666413908634967, 0.5654338242823064, 0.16677678440775978, 0.39633497528643125, 0.5281662695844505, 0.3403493990868437, 0.9284206159278044, 0.7172783284321627, 0.5014698216327682, 0.31389109927396563, 0.35264977504551487, 0.045658681456768324, 0.45343985542565346, 0.9372607363923966, 0.8697954196727156, 0.49538197908930715, 0.4963378716091432, 0.6517608619543365, 0.2687669628769046, 0.43343449066118356, 0.6914900250295085, 0.871962612227946, 0.9282440290458073, 0.44405034807774935, 0.006067876102863634, 0.9956463617252003, 0.929852663999335, 0.8619513137966176, 0.13093519844180512, 0.5617923876824588, 0.45316968742780217, 0.24812198380343287, 0.7029287180288709, 0.019736626336854135, 0.35835395707343964, 0.058995065951774395, 0.5403228315461877, 0.966545257776856, 0.16800584935433094, 0.022093850148500405, 0.2957608162764499, 0.7059265044484518, 0.7808513714271873, 0.4152867350477728, 0.6425485271492697, 0.9763940629034503, 0.6162805622940686, 0.20037591496531848, 0.013004317131549814, 0.10108480919142393, 0.6338934020226543, 0.9811472347156552, 0.5126812289219432, 0.35625397790368396, 0.010027539204480629, 0.242877952977266, 0.7836616383707724, 0.9616574831529631, 0.6736272362122306, 0.6060907434272468, 0.08323435549034452]), [7, 10]));
    assert.ok(checknear(_m.matrix_to32fv(result), [1.4458341107025223, 0.8283675468893227, 1.0675693644097943, 1.002819425205797, 0.9966515689646244, 1.2232589820495563, 1.9177809234654686, 1.2123549056533007, 1.289305287206543, 0.6392375411812009, 1.735755913436215, 1.4653472774514242, 1.1073251650740776, 1.3947782448718167, 1.4784346583586994, 1.9656214843889661, 1.8978901175075742, 0.946984142878458, 0.8720279226148046, 1.1602818970189626, 1.7708311407471948, 1.5032643639749976, 2.1292424934690346, 2.31073850624179, 2.1740957408767465, 2.2222972382139434, 2.7135059239280808, 1.4089362193930595, 1.730591459941252, 1.7338624527270259, 2.1575876683486737, 1.8128345996035848, 1.8716414013631268, 2.326468112969659, 2.3871610855338035, 2.4298147041119504, 2.838604304959189, 1.3858997215335087, 1.6075794050017511, 1.7675645378042828, 2.7865713260703133, 2.535774870639838, 2.397688991358371, 2.3167986408431838, 2.6736835690434355, 3.0175931306504578, 3.664647681310475, 1.770223420215212, 2.147934004491507, 1.9979485904681593, 1.7322895174191701, 1.5966629197289373, 1.0393947498538152, 0.7631965161291187, 1.7985361022501076, 1.7266513742137781, 1.6639970087430738, 0.6010315206211176, 1.0345121584976666, 1.0235397719528478, 1.9290861688846501, 2.165413242574195, 1.9073932348041651, 2.4081082812003545, 2.448437291349138, 2.096722312643222, 2.8637879524467453, 1.1535292961946135, 1.5452009309057408, 1.885276224742847, 1.6695706596431585, 1.2848694638773257, 1.0856946764761144, 0.7228070334002586, 1.4795263818979272, 1.6407657789982468, 1.8429949716661589, 0.8799467661907757, 1.2203941912445202, 0.8389007395843638, 1.7646071090787112, 1.8277980499823925, 1.2257264078148766, 1.2423523833746966, 1.3938562985537748, 2.2009825270387986, 2.595009163063537, 1.063856658003071, 1.122050059128904, 1.1946746769299819, 2.053266727695041, 1.49451599434565, 1.9745733087570025, 1.6788347880321934, 2.039700882927648, 2.067174897687284, 2.7033880282100453, 1.5086330663428973, 1.9468478048721167, 1.36546153783569]));
    console.log("matrix_matrix_mul passed...");

    console.log("All passed!")
})();
