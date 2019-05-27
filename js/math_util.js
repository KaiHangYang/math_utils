/*** Part of the functions are copied and modified from https://github.com/sloisel/numeric/blob/master/src/numeric.js ***/
_m = {};
(function() {
    /******************* Define the mat class *******************/
    function K_Mat() {
        var args = arguments;
        if (args[0] instanceof Float32Array || args[0] instanceof Array) {
            this._size = Math.floor(Math.sqrt(args[0].length));
            this._transpose = false;
            this._arr = new Float32Array(args[0]);
        }
        else if (args[0] instanceof K_Mat) {
            this._transpose = args[0]._transpose;
            this._size = args[0]._size;
            this._arr = new Float32Array(args[0]._arr);
        }
        else {
            this._size = args[0];
            this._transpose = false;
            this._arr = new Float32Array(this._size * this._size);
        }
    }

    K_Mat.prototype.toFloat32Array = function() {
        if (this._transpose) {
            var result = new Float32Array(this._size*this._size);
            for (var i = 0; i < this._size; ++i) {
                for (var j = 0; j < this._size; ++j) {
                    result[i * this._size + j] = this.get(i, j);
                }
            }
            return result;
        }
        else {
            return new Float32Array(this._arr);
        }
    }

    K_Mat.prototype.get = function() {
        if (arguments.length == 2) {
            // i, j
            var i = arguments[0];
            var j = arguments[1];
            if (this._transpose) {
                return this._arr[j * this._size + i];
            }
            else {
                return this._arr[i * this._size + j];
            }
        }
        else {
            // i to get row_val
            var i = arguments[0];
            var result = new Float32Array(this._size);

            for (var j = 0; j < this._size; ++j) {
                if (this._transpose) {
                    result[j] = this._arr[j * this._size + i];
                }
                else {
                    result[j] = this._arr[i * this._size + j];
                }
            }

            return result;
        }
    }
    K_Mat.prototype.set = function() {
        if (arguments.length == 3) {
            // i, j, val
            var i = arguments[0];
            var j = arguments[1];
            var val = arguments[2];
            if (this._transpose) {
                this._arr[j * this._size + i] = val;
            }
            else {
                this._arr[i * this._size + j] = val;
            }
        }
        else {
            // i, row_val
            var i = arguments[0];
            var row_val = arguments[1];

            for (var j = 0; j < this._size; ++j) {
                if (this._transpose) {
                    this._arr[j * this._size + i] = row_val[j];
                }
                else {
                    this._arr[i * this._size + j] = row_val[j];
                }
            }
        }
    }
    K_Mat.prototype.transpose = function() {
        this._transpose = !this._transpose;
    }
    K_Mat.prototype.inplace_fill = function(val) {
        this._transpose = false;
        for (var i = 0; i < this._size * this._size; ++i) {
            this._arr[i] = val;
        }
    }
    K_Mat.prototype.clone = function() {
        return new K_Mat(this);
    }
    K_Mat.prototype.inplace_identity = function() {
        this._transpose = false;
        this.inplace_fill(0);
        for (var i = 0; i < this._size; ++i) {
            this.set(i, i, 1);
        }
    }

    K_Mat.prototype.inverse = function() {
        var A = new K_Mat(this);
        var I = new K_Mat(this);
        I.inplace_identity();

        var abs = Math.abs, m = A._size, n = A._size;
        var Ai, Aj;
        var Ii, Ij;
        var i,j,k,x;

        for(j=0;j<n;++j) {
            var i0 = -1;
            var v0 = -1;
            for(i=j;i!==m;++i) { k = abs(A.get(i, j)); if(k>v0) { i0 = i; v0 = k; } }
            Aj = A.get(i0);
            A.set(i0, A.get(j));
            A.set(j, Aj)
            Ij = I.get(i0);
            I.set(i0, I.get(j));
            I.set(j, Ij);
            x = Aj[j];
            for(k=j;k!==n;++k)  A.set(j, k, A.get(j, k) / x); 
            for(k=n-1;k!==-1;--k) I.set(j, k, I.get(j, k) / x);
            for(i=m-1;i!==-1;--i) {
                if(i!==j) {
                    Ai = A.get(i);
                    Ii = I.get(i);
                    x = Ai[j];
                    for(k=j+1;k!==n;++k)  A.set(i, k, A.get(i, k) - A.get(j, k) * x);
                    for(k=n-1;k>0;--k) {
                        I.set(i, k, I.get(i, k) - I.get(j, k) * x);
                        --k;
                        I.set(i, k, I.get(i, k) - I.get(j, k) * x);
                    }
                    if(k===0) I.set(i, 0, I.get(i, 0) - I.get(j, 0) * x);
                }
            }
        }
        return I;
    }
    K_Mat.prototype.dot = function(elm) {
        if (elm instanceof K_Mat) {
            // matrix matrix dot
            var result = new K_Mat(this._size);
            for (var i = 0; i < this._size; ++i) {
                for (var j = 0; j < this._size; ++j) {
                    result.set(i, j, this.get(i, 0) * elm.get(0, j) + this.get(i, 1) * elm.get(1, j) + this.get(i, 2) * elm.get(2, j) + this.get(i, 3) * elm.get(3, j));
                }
            }
            return result;
        }
        else {
            // matrix vec dot
            elm = new K_Vec(elm);
            var result = new Float32Array(this._size);
            for (var i = 0; i < this._size; ++i) {
                result[i] = elm.dot(this.get(i));
            }

            return result;
        }
    }

    function K_Vec() {
        var args = arguments;
        if (args[0] instanceof Float32Array || args[0] instanceof Array) {
            this._arr = new Float32Array(args[0]);
            this._size = this._arr.length;
        }
        else if (args[0] instanceof K_Vec) {
            this._arr = new Float32Array(args[0]._arr);
            this._size = this._arr.length;
        }
        else {
            this._size = args[0];
            this._arr = new Float32Array(this._size)
        }
    }

    K_Vec.prototype.toFloat32Array = function() {
        return new Float32Array(this._arr);
    }

    K_Vec.prototype.get = function(i) {
        return this._arr[i];
    }
    K_Vec.prototype.set = function(i, val) {
        this._arr[i] = val;
    }

    K_Vec.prototype.clone = function() {
        return new K_Vec(this);
    }
    K_Vec.prototype.norm = function() {
        return Math.sqrt(Array.prototype.map.call(this._arr, function(a) {return a * a;}).reduce(function(a, b) {return a + b;}));
    }
    K_Vec.prototype.normalize = function() {
        var norm = this.norm();
        return new K_Vec(Array.prototype.map.call(this._arr, function(a) {return a / norm;}));
    }
    K_Vec.prototype.inplace_normalize = function() {
        var norm = this.norm();
        for (var i = 0; i < this._size; ++i) {
            this.set(i, this.get(i) / norm);
        }
    }
    K_Vec.prototype.minus = function(vec) {
        var result = new K_Vec(vec);
        for (var i = 0; i < this._size; ++i) {
            result.set(i, this.get(i) - vec.get(i));
        }
        return result;
    }
    K_Vec.prototype.add = function(vec) {
        var result = new K_Vec(vec);
        for (var i = 0; i < this._size; ++i) {
            result.set(i, this.get(i) + vec.get(i));
        }
        return result;
    }
    K_Vec.prototype.dot = function(vec) {
        var sum = 0.0;
        if (vec instanceof K_Vec) {
            for (var i = 0; i < this._size; ++i) {
                sum += this.get(i) * vec.get(i);
            }
        }
        else {
            for (var i = 0; i < this._size; ++i) {
                sum += this.get(i) * vec[i];
            }
        }
        return sum;
    }
    K_Vec.prototype.scale = function(scale) {
        var result = new K_Vec(this._arr);
        for (var i = 0; i < this._size; ++i) {
            result.set(i, this.get(i) * scale);
        }

        return result;
    }
    K_Vec.prototype.inplace_scale = function(scale) {
        for (var i = 0; i < this._size; ++i) {
            this.set(i, this.get(i) * scale);
        }
    }

    K_Vec.prototype.cross = function(vec) {
        // ATTENTION: Only support vector of size 3
        var result = new K_Vec(3);
        result.set(0, this.get(1) * vec.get(2) - this.get(2) * vec.get(1));
        result.set(1, -(this.get(0) * vec.get(2) - this.get(2) * vec.get(0)));
        result.set(2, this.get(0) * vec.get(1) - this.get(1) * vec.get(0));
        return result;
    }
    /******************* ************* *******************/
    function deg2rad(deg) {
        return deg * Math.PI / 180;
    } 
    
    function perspective(fov, ratio, near, far) {
        var tmp = 1.0 / Math.tan(fov/2.0);
        var matrix = new K_Mat([tmp / ratio, 0, 0, 0, 0, tmp, 0, 0, 0, 0, -(far + near) / (far - near), -2 * far * near / (far - near), 0, 0, -1, 0]);
        return matrix;
    }
    
    function lookAt(eye, target, up) {
        eye = new K_Vec(eye);
        target = new K_Vec(target);
        up = new K_Vec(up);
    
        var Z = eye.minus(target).normalize();
        var X = up.cross(Z).normalize();
        var Y = Z.cross(X);

        eye.inplace_scale(-1);
        var matrix = new K_Mat([X.get(0),X.get(1),X.get(2),X.dot(eye), Y.get(0),Y.get(1),Y.get(2),Y.dot(eye), Z.get(0),Z.get(1),Z.get(2),Z.dot(eye), 0., 0., 0., 1.]);
    
        return matrix
    }
    
    function translate(target) {
        return new K_Mat([1, 0, 0, target[0], 0, 1, 0, target[1], 0, 0, 1, target[2], 0, 0, 0, 1]);
    }
    
    function rotate(axis, theta) {
        var axis = new K_Vec(axis);
        axis.inplace_normalize();

        var u = axis.get(0);
        var v = axis.get(1);
        var w = axis.get(2);
        var cos_theta = Math.cos(theta);
        var sin_theta = Math.sin(theta);

        var matrix = new K_Mat([cos_theta + (u * u) * (1 - cos_theta), u * v * (1 - cos_theta) - w * sin_theta, u * w * (1 - cos_theta) + v * sin_theta, 0,
                                u * v * (1 - cos_theta) + w * sin_theta, cos_theta + v * v * (1 - cos_theta), v * w * (1 - cos_theta) - u * sin_theta, 0,
                                u * w * (1 - cos_theta) - v * sin_theta, w * v * (1 - cos_theta) + u * sin_theta, cos_theta + w * w * (1 - cos_theta), 0,
                                0, 0, 0, 1])
    
        return matrix;
    }
    function scale(ratio) {
        return new K_Mat([ratio[0],0,0,0, 0,ratio[1],0,0, 0,0,ratio[2],0, 0,0,0,1]);
    }
    
    _m.deg2rad = deg2rad;
    _m.perspective = perspective;
    _m.lookAt = lookAt;
    _m.translate = translate;
    _m.rotate = rotate;
    _m.scale = scale;
    _m.K_Mat = K_Mat;
    _m.K_Vec = K_Vec;
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

    var result, mat, mat1, mat2;
    /******** Check matrix_inverse *******/
    mat = new _m.K_Mat([0.11555905577932213, 0.738184501960425, 0.5804137287091379, 0.4699458894558518, 0.8095798004366057, 0.0829459627383573, 0.10706353602349172, 0.08305823306043891, 0.4650258290362618, 0.6944323567403545, 0.06483547976151582, 0.014406512798651794, 0.9751298776384586, 0.4517797277610748, 0.41371687160641346, 0.40297826091354716]);
    result = mat.inverse();
    assert.ok(checknear(result.toFloat32Array(), [-0.29389774620637715, 1.0736758580152652, 0.10766733253408481, 0.11759271379069397, -0.4052469760188802, -1.7925825815884169, 1.573695050481297, 0.7858027721702396, 8.018690995387887, 15.066770412605207, -2.272401027873725, -12.375438850473472, -7.066874785593195, -16.056692594278125, 0.30814935207818495, 14.021227601882144]));

    mat = new _m.K_Mat([0.11555905577932213, 0.738184501960425, 0.5804137287091379, 0.4699458894558518, 0.8095798004366057, 0.0829459627383573, 0.10706353602349172, 0.08305823306043891, 0.4650258290362618, 0.6944323567403545, 0.06483547976151582, 0.014406512798651794, 0.9751298776384586, 0.4517797277610748, 0.41371687160641346, 0.40297826091354716]);
    result = mat.inverse();
    assert.ok(checknear(result.toFloat32Array(), [-0.29389774620637715, 1.0736758580152652, 0.10766733253408481, 0.11759271379069397, -0.4052469760188802, -1.7925825815884169, 1.573695050481297, 0.7858027721702396, 8.018690995387887, 15.066770412605207, -2.272401027873725, -12.375438850473472, -7.066874785593195, -16.056692594278125, 0.30814935207818495, 14.021227601882144]));

    mat = new _m.K_Mat([0.5371271935192891, 0.5098596741585567, 0.7045528965017045, 0.7770975383978029, 0.22925940324546168, 0.6352887450889843, 0.1502475800152795, 0.21631949115648696, 0.12088712601438922, 0.014772539263734008, 0.06933264987238963, 0.08435517843804063, 0.5254639946672002, 0.11596844725212663, 0.31210968169002584, 0.5810935455011863]);
    mat.transpose();
    result = mat.inverse();
    assert.ok(checknear(result.toFloat32Array(), [-1.470807690883135, -0.07314633750578929, 2.7246029352464562, -0.11880325218639291, 0.9378495249364549, 1.7271959674139172, -1.5930727219651912, -0.33711090678248384, 16.250881015455303, -1.574383371253796, 11.31505257522917, -20.458343772526124, -0.741292115775163, -0.3166046589095654, -4.69313908859165, 4.975124291960501]));

    mat.transpose();
    result = mat.inverse();
    assert.ok(checknear(result.toFloat32Array(), [-1.4708076908831342, 0.9378495249364551, 16.25088101545532, -0.7412921157751672, -0.07314633750578908, 1.727195967413917, -1.5743833712537958, -0.3166046589095658, 2.724602935246457, -1.5930727219651915, 11.31505257522919, -4.693139088591655, -0.11880325218639376, -0.3371109067824837, -20.458343772526153, 4.975124291960509]));
    console.log("matrix_inverse passed...");
    /*************************************/

    /********** Check perspective *********/
    result = _m.perspective(_m.deg2rad(32.78), 7/3.0, 0.001, 100000);
    assert.ok(checknear(result.toFloat32Array(), [1.457100, 0.000000, 0.000000, 0.000000, 0.000000, 3.399899, 0.000000, 0.000000, 0.000000, 0.000000, -1.000000, -0.002000, 0.000000, 0.000000, -1.000000, 0.000000]));
    console.log("perspective passed...");
    /**************************************/

    /********** Check lookAt *********/
    result = _m.lookAt([1,3, 22], [7, 0, -10], [1, 0, 0]);
    assert.ok(checknear(result.toFloat32Array(), [0.000000, -0.995634, 0.093341, 0.933407, 0.983018, 0.017129, 0.182710, -5.054024, -0.183511, 0.091756, 0.978726, -21.623728, 0.000000, 0.000000, 0.000000, 1.000000]));
    console.log("lookAt passed...");
    /*********************************/

    /********** Check translate *********/
    result = _m.translate([1, -73, 33]);
    assert.ok(checknear(result.toFloat32Array(), [1.000000, 0.000000, 0.000000, 1.000000, 0.000000, 1.000000, 0.000000, -73.000000, 0.000000, 0.000000, 1.000000, 33.000000, 0.000000, 0.000000, 0.000000, 1.000000]));
    console.log("translate passed...");
    /*********************************/
    
    /********** Check rotate *********/
    result = _m.rotate([1, 33, 14], _m.deg2rad(33.78));
    assert.ok(checknear(result.toFloat32Array(), [0.831310, -0.212731, 0.513487, 0.000000, 0.221396, 0.974139, 0.045145, 0.000000, -0.509812, 0.076154, 0.856909, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000]));
    console.log("rotate passed...");
    /*********************************/

    /********** Check scale *********/
    result = _m.scale([0.22, 33, 11]);
    assert.ok(checknear(result.toFloat32Array(), [0.220000, 0.000000, 0.000000, 0.000000, 0.000000, 33.000000, 0.000000, 0.000000, 0.000000, 0.000000, 11.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000]));
    console.log("scale passed...");
    /*********************************/
    mat = new _m.K_Mat(new Float32Array([0.20269068663047796, 0.20767331384190613, 0.40394200937545044, 0.7193801041923747, 0.21706178592760728, 0.46867886083023624, 0.5819174042173045, 0.07943043338986899, 0.9067046937935206, 0.18125923594031768, 0.7941948842091442, 0.6694793052236547, 0.4414798425055976, 0.950857655329681, 0.039357913400368205, 0.33679399542152855]));
    vec = new _m.K_Vec([0.5479187 , 0.91653357, 0.90698666, 0.87934211]);
    assert.ok(checknear(mat.dot(vec), [1.300348812934088, 1.1461299668223979, 1.9719561390552678, 1.4452422637873097]));

    mat.transpose();
    assert.ok(checknear(mat.dot(vec), [1.5205833080838969, 1.5438768859388659, 1.5096074497393475, 1.3703284095201493]));
    mat.transpose();
    assert.ok(checknear(mat.dot(vec), [1.300348812934088, 1.1461299668223979, 1.9719561390552678, 1.4452422637873097]));

    mat = new _m.K_Mat([0.06312506158827758, 0.40223817049604416, 0.8593638803614607, 0.9178191691840003, 0.6763715020221625, 0.2866626647867673, 0.04183105770671858, 0.19590463836729588, 0.7348479584534598, 0.2500269326174769, 0.3195485304108756, 0.11690055129487298, 0.2521069609955625, 0.4116091282768064, 0.3884687148309862, 0.9293582867079737]);
    mat1 = new _m.K_Mat(mat)
    mat2 = new _m.K_Mat([0.5762798098800374, 0.21743186694361516, 0.7276688614291815, 0.9601758112958034, 0.42234718752310463, 0.5449402926211812, 0.338524186116842, 0.3043997069328981, 0.7226416440790925, 0.36659214845613775, 0.4764076482884494, 0.8158889950410532, 0.7435425099202836, 0.355822624412482, 0.8425253551108041, 0.7058084782371443])

    assert.ok(checknear(mat1.dot(mat2).toFloat32Array(), [1.5097115545888171, 0.8745380630480387, 1.3647949376971826, 1.5320024222611743, 0.6867427016655718, 0.3883209947399326, 0.7741999869023736, 0.9090962411606448, 0.8469158182843413, 0.45476895660631617, 0.8700931835024464, 1.1249168892594992, 1.2908671731113202, 0.7522147713812636, 1.2908674178142512, 1.3402570112523766]));
    mat1.transpose();
    assert.ok(checknear(mat1.dot(mat2).toFloat32Array(), [1.040525279482585, 0.7414023365968118, 0.8373959485033402, 1.0439920370407643, 0.8396016646563067, 0.48179118345513366, 0.8556443067805706, 0.967990828083915, 1.0326633617961345, 0.4650184616308846, 1.1190232669400506, 1.3727744145323657, 1.3871550290543084, 0.6798609952808494, 1.5728871253576198, 1.6922279112940415]));
    mat2.transpose();
    assert.ok(checknear(mat1.dot(mat2).toFloat32Array(), [0.9602353999815155, 0.72074786855449, 0.8493477631554607, 1.0846717173012783, 0.871285276871904, 0.5360320583458992, 0.8507044360015246, 0.9022534832302498, 1.109853237287202, 0.6121700158862837, 1.1055297779189113, 1.1972702647911055, 1.5489288055702148, 0.8168647298106505, 1.5490169709592796, 1.5065855078372465]));
    console.log("Matrix vector fn passed...")
    console.log("All passed!")

    mat = new _m.K_Mat([0.5656522430408937, 0.11833214994732077, 0.5875166796918702, 0.3629238310241295, 0.7156402018599957, 0.8930635945356458, 0.6561836501973131, 0.2010850282642005, 0.9844428705775882, 0.12022961729651382, 0.45721989282278686, 0.6082416610396965, 0.5575818039675446, 0.1584998965469563, 0.5604962456184088, 0.2654241215196801])
    mat1 = new _m.K_Mat(mat);
    mat1.set(2, [0.43502239, 0.87532924, 0.79660435, 0.56192567]);
    mat1.set(3, 1, 1234.134134);
    assert.ok(checknear(mat1.toFloat32Array(), [0.5656522430408937, 0.11833214994732077, 0.5875166796918702, 0.3629238310241295, 0.7156402018599957, 0.8930635945356458, 0.6561836501973131, 0.2010850282642005, 0.43502238860464504, 0.8753292384412149, 0.7966043456993025, 0.5619256684983286, 0.5575818039675446, 1234.134134, 0.5604962456184088, 0.2654241215196801]));
    assert.ok(checknear(mat.toFloat32Array(), [0.5656522430408937, 0.11833214994732077, 0.5875166796918702, 0.3629238310241295, 0.7156402018599957, 0.8930635945356458, 0.6561836501973131, 0.2010850282642005, 0.9844428705775882, 0.12022961729651382, 0.45721989282278686, 0.6082416610396965, 0.5575818039675446, 0.1584998965469563, 0.5604962456184088, 0.2654241215196801]));

})();
